#include "duckdb/execution/operator/set/physical_recursive_cte.hpp"

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/aggregate_hashtable.hpp"
#include "duckdb/execution/executor.hpp"
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/parallel/event.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/storage/buffer_manager.hpp"

#include <utility>
#include <iostream>
#include <fstream>
namespace duckdb {

PhysicalRecursiveCTE::PhysicalRecursiveCTE(PhysicalPlan &physical_plan, string ctename, idx_t table_index,
                                           vector<LogicalType> types, bool union_all, PhysicalOperator &top,
                                           PhysicalOperator &bottom, idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::RECURSIVE_CTE, std::move(types), estimated_cardinality),
      ctename(std::move(ctename)), table_index(table_index), union_all(union_all) {
	children.push_back(top);
	children.push_back(bottom);
}

PhysicalRecursiveCTE::~PhysicalRecursiveCTE() {
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class RecursiveCTEState : public GlobalSinkState {
public:
	explicit RecursiveCTEState(ClientContext &context, const PhysicalRecursiveCTE &op)
	    : intermediate_table(context, op.GetTypes()), new_groups(STANDARD_VECTOR_SIZE) {

		vector<BoundAggregateExpression *> payload_aggregates_ptr;
		for (idx_t i = 0; i < op.payload_aggregates.size(); i++) {
			auto &dat = op.payload_aggregates[i];
			payload_aggregates_ptr.push_back(dat.get());
		}

		ht = make_uniq<GroupedAggregateHashTable>(context, BufferAllocator::Get(context), op.distinct_types,
		                                          op.payload_types, payload_aggregates_ptr);

		// MODIFICATION: Use the actual flags from the operator instead of hardcoded value
		use_aggregation = op.use_min_key || op.use_max_key;
	}

	unique_ptr<GroupedAggregateHashTable> ht;

	mutex intermediate_table_lock;
	ColumnDataCollection intermediate_table;
	ColumnDataScanState scan_state;
	bool initialized = false;
	bool finished_scan = false;
	SelectionVector new_groups;
	AggregateHTScanState ht_scan_state;
	// MODIFICATION
	// MODIFICATION: Replace the hardcoded boolean with actual state
	bool has_converged = false;
	bool use_aggregation = false;  // Whether to use MIN/MAX aggregation

	// END

};

unique_ptr<GlobalSinkState> PhysicalRecursiveCTE::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<RecursiveCTEState>(context, *this);
}

idx_t PhysicalRecursiveCTE::ProbeHT(DataChunk &chunk, RecursiveCTEState &state) const {
	Vector dummy_addresses(LogicalType::POINTER);

	// Use the HT to eliminate duplicate rows
	idx_t new_group_count = state.ht->FindOrCreateGroups(chunk, dummy_addresses, state.new_groups);

	// we only return entries we have not seen before (i.e. new groups)
	chunk.Slice(state.new_groups, new_group_count);

	return new_group_count;
}

void PopulateChunk(DataChunk &group_chunk, DataChunk &input_chunk, const vector<idx_t> &idx_set, bool reference) {
	idx_t chunk_index = 0;
	// Populate the group_chunk
	for (auto &group_idx : idx_set) {
		if (reference) {
			// Reference from input_chunk[chunk_index] -> group_chunk[group_idx]
			group_chunk.data[chunk_index++].Reference(input_chunk.data[group_idx]);
		} else {
			// Reference from input_chunk[group.index] -> group_chunk[chunk_index]
			group_chunk.data[group_idx].Reference(input_chunk.data[chunk_index++]);
		}
	}
	group_chunk.SetCardinality(input_chunk.size());
}
// Call this right after your FindOrCreate/AddChunk logic,

static inline void PrintKeysAddrsNewGroups(
    duckdb::DataChunk &distinct_rows,
    duckdb::Vector &addresses,                    // from FindOrCreateGroups(...)
    const duckdb::SelectionVector &new_groups,    // filled by FindOrCreateGroups
    duckdb::idx_t new_groups_count,
    const char *path = "test_debug_ht.txt") {

    FILE *fp = std::fopen(path, "a");
    if (!fp) return;

    const duckdb::idx_t n = distinct_rows.size();
    const duckdb::idx_t kcols = distinct_rows.ColumnCount();

    // Ensure addresses is flat before reading
    addresses.Flatten(n);

    using namespace duckdb;
    const auto addr_type = addresses.GetType().id();
    const auto &validity = FlatVector::Validity(addresses);

    // Helper lambdas
    auto print_key_row = [&](idx_t r) {
        std::fprintf(fp, "key=[");
        for (idx_t c = 0; c < kcols; c++) {
            if (c) std::fprintf(fp, ", ");
            auto s = distinct_rows.GetValue(c, r).ToString();
            std::fprintf(fp, "%s", s.c_str());
        }
        std::fprintf(fp, "] ");
    };

    auto print_addr = [&](idx_t r) {
        if (!validity.RowIsValid(r)) {
            std::fprintf(fp, "addr=NULL\n");
            return;
        }
        switch (addr_type) {
        case LogicalTypeId::POINTER: {
            auto data = FlatVector::GetData<data_ptr_t>(addresses);
            std::fprintf(fp, "addr=%p\n", (void*)data[r]);
            break;
        }
        case LogicalTypeId::UBIGINT: {
            auto data = FlatVector::GetData<uint64_t>(addresses);
            std::fprintf(fp, "addr=0x%llx\n", (unsigned long long)data[r]);
            break;
        }
        case LogicalTypeId::BIGINT: {
            auto data = FlatVector::GetData<int64_t>(addresses);
            std::fprintf(fp, "addr=%lld (0x%llx)\n",
                         (long long)data[r], (unsigned long long)data[r]);
            break;
        }
        default: {
            // Fallback: still safe, but less precise
            auto s = addresses.GetValue(r).ToString();
            std::fprintf(fp, "addr=%s\n", s.c_str());
            break;
        }
        }
    };

    // Print keys + addresses row-aligned
    for (duckdb::idx_t r = 0; r < n; r++) {
        print_key_row(r);
        print_addr(r);
    }

    // Print the new_groups selection
    std::fprintf(fp, "new_groups_indices=[");
    for (duckdb::idx_t i = 0; i < new_groups_count; i++) {
        if (i) std::fprintf(fp, ", ");
        std::fprintf(fp, "%u", (unsigned)new_groups.get_index(i));
    }
    std::fprintf(fp, "]\n");

    std::fclose(fp);
}
SinkResultType PhysicalRecursiveCTE::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<RecursiveCTEState>();

	lock_guard<mutex> guard(gstate.intermediate_table_lock);
	// // Debug: Print incoming chunk
	// printf("=== INCOMING CHUNK ===\n");
	// for (idx_t row = 0; row < chunk.size(); row++) {
	// 	printf("Row %llu: x=%s, c=%s\n",
	// 	       (unsigned long long)row,
	// 	       chunk.GetValue(0, row).ToString().c_str(),
	// 	       chunk.GetValue(1, row).ToString().c_str());
	// }
	// printf("======================\n");

	if (!using_key) {


		if (!union_all) {
			idx_t match_count = ProbeHT(chunk, gstate);
			if (match_count > 0) {
				gstate.intermediate_table.Append(chunk);


			}
		} else {

			gstate.intermediate_table.Append(chunk);

		}
	} else {
		// Split incoming DataChunk into payload and keys
		DataChunk distinct_rows;
		distinct_rows.Initialize(Allocator::DefaultAllocator(), distinct_types);
		PopulateChunk(distinct_rows, chunk, distinct_idx, true);

		DataChunk payload_rows;
		if (!payload_types.empty()) {
			payload_rows.Initialize(Allocator::DefaultAllocator(), payload_types);
		}
		PopulateChunk(payload_rows, chunk, payload_idx, true);
		// Debug: Print what we're about to process
		// printf("=== PROCESSING KEYS/PAYLOAD ===\n");
		// for (idx_t row = 0; row < distinct_rows.size(); row++) {
		// 	printf("Key row %llu: x=%s, payload c=%s\n",
		// 	       (unsigned long long)row,
		// 	       distinct_rows.GetValue(0, row).ToString().c_str(),
		// 	       payload_rows.size() > 0 ? payload_rows.GetValue(0, row).ToString().c_str() : "N/A");
		// }
		// printf("===============================\n");

		//MODIFICATION
		// Use FindOrCreateGroups to detect new groups and get addresses
			// OLD IMPLEMENTATION OF ITERATION
			// idx_t new_group_count = 0;
			// if (gstate.min_function) {
			// 	Vector addresses(LogicalType::POINTER);
			// 	new_group_count = gstate.ht->FindOrCreateGroups(distinct_rows, addresses, gstate.new_groups);

			//
			// }
		idx_t new_group_count = 0;
		bool has_updates = false;
		Vector addresses(LogicalType::POINTER);
		if (gstate.use_aggregation) {
            new_group_count = gstate.ht->FindOrCreateGroups(distinct_rows, addresses, gstate.new_groups);

            // Get current aggregated values BEFORE adding the chunk
            DataChunk before_values;
            before_values.Initialize(Allocator::DefaultAllocator(), payload_types);
            gstate.ht->FetchAggregates(distinct_rows, before_values);
            
            // Add the chunk (this will update MIN values if better)
            gstate.ht->AddChunk(distinct_rows, payload_rows, AggregateType::NON_DISTINCT);
            
            // Get aggregated values AFTER adding the chunk
            DataChunk after_values;
            after_values.Initialize(Allocator::DefaultAllocator(), payload_types);
            gstate.ht->FetchAggregates(distinct_rows, after_values);
            
            has_updates = false;
            for (idx_t row = 0; row < distinct_rows.size(); row++) {
                for (idx_t col = 0; col < payload_types.size(); col++) {
                    auto before_val = before_values.GetValue(col, row);
                    auto after_val = after_values.GetValue(col, row);

                    if (Value::NotDistinctFrom(before_val, after_val) == false) {
                        has_updates = true;
                        // printf("MIN update detected for x=%s: %s -> %s\n",
                        //        distinct_rows.GetValue(0, row).ToString().c_str(),
                        //        before_val.ToString().c_str(),
                        //        after_val.ToString().c_str());
                        break;
                    }
                }
                if (has_updates) break;
            }
            
            if (has_updates) {
                gstate.intermediate_table.Append(chunk);
            }

        }

		//END
		// Add the chunk to the hash table and append it to the intermediate table
		gstate.ht->AddChunk(distinct_rows, payload_rows, AggregateType::NON_DISTINCT);

		// init result chunk with the saved types (no 'op' here)
		auto &alloc = duckdb::Allocator::DefaultAllocator();
		duckdb::DataChunk agg_values;
		agg_values.Initialize(alloc, payload_types);

		// fetch aggregates for these keys
		gstate.ht->FetchAggregates(distinct_rows, agg_values);

		// // write keys + values to a file (debug only)
		// FILE *fp = fopen("test_debug_ht.txt", "a");
		// if (fp) {
		// 	for (duckdb::idx_t r = 0; r < distinct_rows.size(); r++) {
		// 		// keys
		// 		fprintf(fp, "key=[");
		// 		for (duckdb::idx_t c = 0; c < distinct_rows.ColumnCount(); c++) {
		// 			if (c) fprintf(fp, ", ");
		// 			auto s = distinct_rows.GetValue(c, r).ToString();
		// 			fprintf(fp, "%s", s.c_str());
		// 		}
		// 		// values
		// 		fprintf(fp, "] values=[");
		// 		for (duckdb::idx_t c = 0; c < agg_values.ColumnCount(); c++) {
		// 			if (c) fprintf(fp, ", ");
		// 			auto s = agg_values.GetValue(c, r).ToString();
		// 			fprintf(fp, "%s", s.c_str());
		// 		}
		// 		fprintf(fp, "]\n");
		// 	}
		// 	fprintf(fp, "############################\n");
		// 	fclose(fp);
		// }
		// I COMMENTED THIS TO MODIFY THE INTERMIDATE TABLE WITH ONLY NEW VALUES

		if (gstate.use_aggregation) {
			// Check if new entries were added to the hash table
			// OLD IMPLEMENTATION
			// if (gstate.min_function && new_group_count > 0) {
			// 	gstate.has_converged = false;
			// }
			// Only append rows that were actually inserted or updated in the hash table

			if (has_updates) {
				// Create a filtered chunk containing only the rows that caused updates
				DataChunk filtered_chunk;
				filtered_chunk.Initialize(Allocator::DefaultAllocator(), chunk.GetTypes());
				filtered_chunk.Slice(chunk, gstate.new_groups, new_group_count);

				filtered_chunk.Slice(chunk, gstate.new_groups, new_group_count);



				gstate.intermediate_table.Append(filtered_chunk);
			}
		}
		else {
			gstate.intermediate_table.Append(chunk);

		}



		// END

	}

	return SinkResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
SourceResultType PhysicalRecursiveCTE::GetData(ExecutionContext &context, DataChunk &chunk,
                                               OperatorSourceInput &input) const {
	auto &gstate = sink_state->Cast<RecursiveCTEState>();
	if (!gstate.initialized) {
		if (!using_key) {
			gstate.intermediate_table.InitializeScan(gstate.scan_state);
		} else {
			gstate.ht->InitializeScan(gstate.ht_scan_state);
			recurring_table->InitializeScan(gstate.scan_state);
		}
		gstate.finished_scan = false;
		gstate.initialized = true;


	}

	while (chunk.size() == 0) {
		if (!gstate.finished_scan) {
			if (!using_key) {
				// scan any chunks we have collected so far
				gstate.intermediate_table.Scan(gstate.scan_state, chunk);
			}
			if (chunk.size() == 0) {
				gstate.finished_scan = true;
			} else {
				break;
			}
		} else {
			//MODIFICATION
			// Check for convergence before proceeding with recursion
			// if (gstate.min_function && gstate.has_converged) {
			// 	// We've converged, terminate early
			// 	gstate.finished_scan = true;
			// 	if (using_key) {
			// 		// Extract final results from hash table
			// 		DataChunk payload_rows;
			// 		DataChunk distinct_rows;
			// 		distinct_rows.Initialize(Allocator::DefaultAllocator(), distinct_types);
			// 		if (!payload_types.empty()) {
			// 			payload_rows.Initialize(Allocator::DefaultAllocator(), payload_types);
			// 		}
			//
			// 		gstate.ht->Scan(gstate.ht_scan_state, distinct_rows, payload_rows);
			// 		PopulateChunk(chunk, distinct_rows, distinct_idx, false);
			// 		PopulateChunk(chunk, payload_rows, payload_idx, false);
			// 	}
			// 	break;
			// }
			//END
			// we have run out of chunks
			// now we need to recurse
			// we set up the working table as the data we gathered in this iteration of the recursion

			// After an iteration, we reset the recurring table
			// and fill it up with the new hash table rows for the next iteration.
			if (using_key && ref_recurring && gstate.intermediate_table.Count() != 0) {
				recurring_table->Reset();
				AggregateHTScanState scan_state;
				gstate.ht->InitializeScan(scan_state);

				// Initialise the DataChunks to read the resulting rows.
				// One DataChunk for the payload, one for the keys.
				// Create a new DataChunk to store the result.
				DataChunk result;
				DataChunk payload_rows;
				DataChunk distinct_rows;
				distinct_rows.Initialize(Allocator::DefaultAllocator(), distinct_types);
				if (!payload_types.empty()) {
					payload_rows.Initialize(Allocator::DefaultAllocator(), payload_types);
				}
				result.Initialize(Allocator::DefaultAllocator(), chunk.GetTypes());

				while (gstate.ht->Scan(scan_state, distinct_rows, payload_rows)) {
					// Populate the result DataChunk with the keys and the payload.
					PopulateChunk(result, distinct_rows, distinct_idx, false);
					PopulateChunk(result, payload_rows, payload_idx, false);
					// Append the result to the recurring table.
					recurring_table->Append(result);
				}
			}

			working_table->Reset();
			working_table->Combine(gstate.intermediate_table);

			//MODIFICATION

			// Set convergence flag to true before recursion
			// It will be reset to false in Sink() if new data is added
			// gstate.has_converged = true;

			//END

			// and we clear the intermediate table
			gstate.finished_scan = false;
			gstate.intermediate_table.Reset();
			// now we need to re-execute all of the pipelines that depend on the recursion
			ExecuteRecursivePipelines(context);


			// check if we obtained any results
			// if not, we are done

			if (gstate.intermediate_table.Count() == 0) {
				gstate.finished_scan = true;
				if (using_key) {
					// Initialise the DataChunks to read the ht.
					// One DataChunk for payload, one for keys.
					DataChunk payload_rows;
					DataChunk distinct_rows;
					distinct_rows.Initialize(Allocator::DefaultAllocator(), distinct_types);
					if (!payload_types.empty()) {
						payload_rows.Initialize(Allocator::DefaultAllocator(), payload_types);
					}

					gstate.ht->Scan(gstate.ht_scan_state, distinct_rows, payload_rows);
					PopulateChunk(chunk, distinct_rows, distinct_idx, false);
					PopulateChunk(chunk, payload_rows, payload_idx, false);
				}
				break;
			}
			if (!using_key) {
				// set up the scan again
				gstate.intermediate_table.InitializeScan(gstate.scan_state);
			}
		}
	}

	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

void PhysicalRecursiveCTE::ExecuteRecursivePipelines(ExecutionContext &context) const {
	if (!recursive_meta_pipeline) {
		throw InternalException("Missing meta pipeline for recursive CTE");
	}
	D_ASSERT(recursive_meta_pipeline->HasRecursiveCTE());

	// get and reset pipelines
	vector<shared_ptr<Pipeline>> pipelines;
	recursive_meta_pipeline->GetPipelines(pipelines, true);
	for (auto &pipeline : pipelines) {
		auto sink = pipeline->GetSink();
		if (sink.get() != this) {
			sink->sink_state.reset();
		}
		for (auto &op_ref : pipeline->GetOperators()) {
			auto &op = op_ref.get();
			op.op_state.reset();
		}
		pipeline->ClearSource();
	}

	// get the MetaPipelines in the recursive_meta_pipeline and reschedule them
	vector<shared_ptr<MetaPipeline>> meta_pipelines;
	recursive_meta_pipeline->GetMetaPipelines(meta_pipelines, true, false);
	auto &executor = recursive_meta_pipeline->GetExecutor();
	vector<shared_ptr<Event>> events;
	executor.ReschedulePipelines(meta_pipelines, events);

	while (true) {
		executor.WorkOnTasks();
		if (executor.HasError()) {
			executor.ThrowException();
		}
		bool finished = true;
		for (auto &event : events) {
			if (!event->IsFinished()) {
				finished = false;
				break;
			}
		}
		if (finished) {
			// all pipelines finished: done!
			break;
		}
	}
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//

static void GatherColumnDataScans(const PhysicalOperator &op, vector<const_reference<PhysicalOperator>> &delim_scans) {
	if (op.type == PhysicalOperatorType::DELIM_SCAN || op.type == PhysicalOperatorType::CTE_SCAN) {
		delim_scans.push_back(op);
	}
	for (auto &child : op.children) {
		GatherColumnDataScans(child, delim_scans);
	}
}

void PhysicalRecursiveCTE::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	op_state.reset();
	sink_state.reset();
	recursive_meta_pipeline.reset();

	auto &state = meta_pipeline.GetState();
	state.SetPipelineSource(current, *this);

	auto &executor = meta_pipeline.GetExecutor();
	executor.AddRecursiveCTE(*this);

	// the LHS of the recursive CTE is our initial state
	auto &initial_state_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
	initial_state_pipeline.Build(children[0]);

	// the RHS is the recursive pipeline
	recursive_meta_pipeline = make_shared_ptr<MetaPipeline>(executor, state, this);
	recursive_meta_pipeline->SetRecursiveCTE();
	recursive_meta_pipeline->Build(children[1]);

	vector<const_reference<PhysicalOperator>> ops;
	GatherColumnDataScans(children[1], ops);

	for (auto op : ops) {
		auto entry = state.cte_dependencies.find(op);
		if (entry == state.cte_dependencies.end()) {
			continue;
		}
		// this chunk scan introduces a dependency to the current pipeline
		// namely a dependency on the CTE pipeline to finish
		auto cte_dependency = entry->second.get().shared_from_this();
		current.AddDependency(cte_dependency);
	}
}

vector<const_reference<PhysicalOperator>> PhysicalRecursiveCTE::GetSources() const {
	return {*this};
}

InsertionOrderPreservingMap<string> PhysicalRecursiveCTE::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["CTE Name"] = ctename;
	result["Table Index"] = StringUtil::Format("%llu", table_index);
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

} // namespace duckdb