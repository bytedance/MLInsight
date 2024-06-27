/*
@author: Steven (Jiaxun) Tang <jtang@umass.edu>
*/
#ifndef MLINSIGHT_FLAMEGRAPH_H
#define MLINSIGHT_FLAMEGRAPH_H

#include <cstdio>
#include <common/MemoryHeap.h>
#include <unordered_map>
#include "trace/hook/HookInstaller.h"
#include "trace/hook/PyHook.h"
#include "trace/tool/AtomicSpinLock.h"
#include "common/TensorObj.h"
#include "analyse/CallBackInterface.h"
#include <regex>

namespace mlinsight {


    typedef ssize_t SnapshotID;

    template<typename NodeType>
    class FlameTree {
    public:
        /**
         * Adding data to flamegraph
         */
        template<typename ...Args>
        void addData(Args...) {
            fatalError("This is a template. Please use subclass.");
        }


        void rmData(NodeType *nodeAddress) {
            fatalError("This is a template. Please use subclass.");
        }

        /**
         * Take a snapshot of the current flamegraph
         * @return snapshotId
         */
        SnapshotID snapshot() {
            fatalError("This is a template. Please use subclass.");
            return -1;
        }

        NodeType *getSnapshot(const SnapshotID &snapshotId) {
            fatalError("This is a template. Please use subclass.");
            return nullptr;
        }


        /**
         * Save all snapshots to file
         */
        void saveToFile() {
            fatalError("This is a template. Please use subclass.");
        }
    };

    typedef ssize_t NODE_ID; //The id of this driverMemRecord node
    class FTMemCell {
    public:
        FTMemCell(const SnapshotID &snapshotId) : snapshotId(snapshotId) {

        }

        /**
          * Copy and swap idiom: copy constructor.
          */
        FTMemCell(const FTMemCell &rho) {
            this->children = rho.children;
            this->allocationSize = rho.allocationSize;
            this->pyModuleId = rho.pyModuleId;
            this->snapshotId = rho.snapshotId;
            this->nextSnapshotNode = rho.nextSnapshotNode;
        }

        /**
         * A special copy constructor that specifies the copied object as the next snapshot version
         * @param alwaysTrueVariable A variabled used to distinguish this copy contrustor from the standard one.
         */
        FTMemCell(FTMemCell &rho, SnapshotID newSnapshotId) : FTMemCell(rho) {
            rho.nextSnapshotNode = this;
            this->snapshotId = newSnapshotId;
        }

        /**
        * Copy and swap idiom: move constructor.
        */
        FTMemCell(FTMemCell &&rho) {
            this->children = std::move(rho.children);
            this->allocationSize = rho.allocationSize;
            this->pyModuleId = rho.pyModuleId;
            this->snapshotId = rho.snapshotId;
            this->nextSnapshotNode = rho.nextSnapshotNode;
        }

        /**
        * Copy and swap idiom: copy assignment.
        */
        FTMemCell &operator=(const FTMemCell &rho) {
            FTMemCell tempObject(rho);
            swap(*this, tempObject);
            return *this;
        }

        /**
         * Copy and swap idiom: move assignment.
         */
        FTMemCell &operator=(FTMemCell &&rho) {
            swap(*this, rho);
            return *this;
        }

        /**
         * Copy and swap idiom: move assignment.
         */
        void swap(FTMemCell &lho, FTMemCell &rho) {
            std::swap(lho.allocationSize, rho.allocationSize);
            std::swap(lho.children, rho.children);
            std::swap(lho.snapshotId, rho.snapshotId);
            std::swap(lho.pyModuleId, rho.pyModuleId);
            std::swap(lho.nextSnapshotNode, rho.nextSnapshotNode);
        }


        inline bool isLeaf() {
            return children.size() == 0;
        }

        friend class FlameTree<FTMemCell>;

    protected:
        /*
        * This node structure must not include a parent pointer.
        * In the copy-on-write implementation, we aim to only update nodes that have changed and keeps the remaining nodes unchanged.
        * A node must be copied if any fields has changed (including driverMemRecord amount, and child references.) Otherwise, old snapshots cannot be recovered.
        * Maintaining a parent reference means that the entire tree has to be copied again at every snapshot.
        * Without a parent reference, only the update path needs to be updated and nodes not on the update path can be reused.
        *
        * However, without a parent reference. It is not very easy to trace back to the root node at the output time.
        * So we keep one outputParent field which is only maintained at the output time but is unmaintained at the snapshot time.
        */
        FTMemCell *outputParent = nullptr;
        std::unordered_map<NODE_ID, FTMemCell *> children; //Child nodes
        ssize_t allocationSize = 0; //The allocation size of this callstack on the flameTree.
        ssize_t pyModuleId = -1;
        SnapshotID snapshotId = 0; //Snapshots are read-only. This field marks the snapshot this cell correlates to.
        FTMemCell *nextSnapshotNode = nullptr; //A linked list that allows tracing from the oldest node to the latest snapshot node version.
    };


    struct FlameGraphSnapshot {
        //todo: investigate whether this will cause race condition when accessing globalExecutionState
        ExecutionState pyTorchModuleStack;
        FTMemCell *snapshotRoot;

        inline FlameGraphSnapshot(const ExecutionState &pytorchModuleStack, FTMemCell *snapshotRoot) :
                pyTorchModuleStack(pytorchModuleStack),
                snapshotRoot(snapshotRoot) {
        }
    };

    //todo: Refactor
    extern std::vector<RegexAndFriendlyName> pyModuleSummaryAttributionArray;

    /**
     * For the flame graph, tree structure decodes.
     * This class is not thread safe. Beware not to update/update at the same time.
     * FlameTree has snapshot capabilities.
     */
    template<>
    class FlameTree<FTMemCell> {
    public:

        FlameTree<FTMemCell>() {
            //Insert root node
            FTMemCell *newRootNode = nodePool.alloc();
            new(newRootNode) FTMemCell(curSnapShotVersion);
            snapShotArray.emplace_back(ExecutionState(-1, PyTorchModuleState::UNSPECIFIED_PYTORCH_MODULE_STATE),
                                       newRootNode);
            //Prevent this object from being freed
            aggregationByPyModule = new std::vector<ssize_t>();
        }

        /**
         * Move to the record stack based on
         * @param stack
         * @return
         */
        FTMemCell *moveToRecordNode(const PyCallStack &stack) {
            //Step1: Move to the child node and copy-on-write.
            //todo: Correctly handle the empty pystack case, currently it is recorded at the root node
            FTMemCell *recordNode = snapShotArray.back().snapshotRoot; // Set current node to the root node.
            assert(recordNode->snapshotId == curSnapShotVersion); // The root node is always created at snapshot time.
            for (ssize_t i = stack.array.size() - 1; i >= 0; --i) {
                //Walk from the stack top to the stack bottom.
                FTMemCell *curChildNode = recordNode->children[stack.array[i].extra->globalPyModuleId];
                if (curChildNode == nullptr) {
                    //Allocate a new node
                    curChildNode = nodePool.alloc();
                    new(curChildNode) FTMemCell(curSnapShotVersion);
                    curChildNode->pyModuleId = stack.array[i].extra->globalPyModuleId;
                    recordNode->children[stack.array[i].extra->globalPyModuleId] = curChildNode;
                } else if (curChildNode->snapshotId < curSnapShotVersion) {
                    // Node exists previously, but belongs to the previous snapshot.
                    //Deep copy this node and update parent-child pointer
                    FTMemCell *curChildNodeCpy = nodePool.alloc();
                    new(curChildNodeCpy) FTMemCell(*curChildNode); //Copy the node and set the nextSnapshot version.
                    curChildNode->nextSnapshotNode = curChildNodeCpy;
                    curChildNodeCpy->snapshotId = curSnapShotVersion;

                    assert(recordNode->snapshotId ==
                           curSnapShotVersion); //The parent node must be at the current snapshot (a.k.a has been updated before).
                    //Re-link the updated parent node to the copied child node
                    recordNode->children[stack.array[i].extra->globalPyModuleId] = curChildNodeCpy;
                    curChildNode = curChildNodeCpy;
                }

                //Move to next layer
                recordNode = curChildNode;
            }
            return recordNode;
        }


        /**
         * Adding data to the flamegraph
         */
        FTMemCell const *addData(const PyCallStack &stack, ssize_t allocationSize) {
            //Move to current node
            FTMemCell *curNode = moveToRecordNode(stack);

            //Update driverMemRecord for the currentNode.
            curNode->allocationSize += allocationSize;
            return curNode;
        }

        /**
         * Update the node according to the allocation size.
         *
         * @param retNodePtr A reference to node address. Be ware that this pointer will be modified to the latest snapshot version.
         * @param stack
         * @param allocationSize
         */
        void rmData(FTMemCell const *&retNodePtr, const PyCallStack &stack,
                    ssize_t allocationSize) {
            FTMemCell *&curNode = const_cast<FTMemCell *&>(retNodePtr); //Force constancy removal
            //Check whether we can take a shortcut. If the node pointed by retNodePtr is in the latest snapshot version. Then we do not need to walk the callstack, but can direct trace to parent node using pointers.
            while (curNode->nextSnapshotNode != nullptr) {
                curNode = curNode->nextSnapshotNode;
            }
            assert(retNodePtr == curNode); //Return the latest snapshot node version.

            //Check whether it is necessary to walk the stack.
            if (curNode->snapshotId == curSnapShotVersion) {
                //This node is already updated to the current snapshot. Update directly and there is no need to walk the stack to create new snapshot nodes.
                curNode->allocationSize -= allocationSize;
            } else {
                //Walk the stack from top to bottom to create new recording nodes for the current snapshot.
                curNode = moveToRecordNode(stack);
                curNode->allocationSize -= allocationSize;
                //Update retNodePtr to points to the latest node version
                retNodePtr = curNode;
            }
        }

        SnapshotID snapshot(const ExecutionState &execState) {
            curSnapShotVersion += 1;
            FTMemCell *previousSnapshotRoot = snapShotArray.back().snapshotRoot;
            //Copy the root node and create a new one
            FTMemCell *curSnapshotRoot = nodePool.alloc();
            new(curSnapshotRoot) FTMemCell(*previousSnapshotRoot);
            curSnapshotRoot->snapshotId = curSnapShotVersion;
            previousSnapshotRoot->nextSnapshotNode = curSnapshotRoot;
            snapShotArray.emplace_back(execState, curSnapshotRoot);
            return curSnapShotVersion - 1;
        }

        const FTMemCell &getSnapshot(const SnapshotID &snapshotId) {
            return (const FTMemCell &) snapShotArray[snapshotId];
        }

        /**
         * Save flame graph to disk for plotting
         */
        void saveToFile(std::ostream &summaryOutput) {
            if (snapShotArray.size() <= 1) {
                return;
            }
            summaryOutput << "Memory Snapshot ================>" << std::endl;
            for (SnapshotID i = 1; i < snapShotArray.size(); ++i) {
                char snapshotFileName[PATH_MAX];
                //summaryOutput << "Snapshot:" << i << std::endl;
                sprintf(snapshotFileName, "%s/memorysnapshot_%zd.folded", logProcessRootPath.c_str(), i);

                summaryOutput << "FlameGraph for snapshot " << i << " saved to:" << snapshotFileName << std::endl;

                FILE * flameGraphOutput = fopen(snapshotFileName, "w");
                if (!flameGraphOutput) {
                    fatalErrorS("Cannot open file %s for writing because: %s", snapshotFileName, strerror(errno));
                }

                //temporary measure. This should be a file.
                saveToFile(i, summaryOutput, flameGraphOutput);
                fclose(flameGraphOutput);
            }
            summaryOutput << "<============= Memory Snapshot" << std::endl;

        }

    private:
        //todo: driverMemRecord leak
        std::vector<ssize_t> *aggregationByPyModule = nullptr; //This struct not be freed by destructor. Otherwise it may be freed before saveToFile is invoked.
    public:
        void saveToFile(SnapshotID snapshotId, std::ostream &summaryOutput, FILE *flameGraphOutput) {

            assert(hookInstallerInstance != nullptr);

            //Make aggregationByPyModule the same size as
            for (ssize_t i = 0; i < aggregationByPyModule->size(); ++i) {
                (*aggregationByPyModule)[i] = 0;
            }
            while (aggregationByPyModule->size() < pyModuleSummaryAttributionArray.size()) {
                aggregationByPyModule->emplace_back(0);
            }

            donotcareModules = 0;

            //A stack used for DFS
            std::vector<FTMemCell *> dfsStack;
            //A stack used for output
            std::vector<FTMemCell *> outputStack;

            //insert root node
            FlameGraphSnapshot &snapshotInfo = snapShotArray[snapshotId];
            FTMemCell *rootNode = snapshotInfo.snapshotRoot;
            rootNode->outputParent = nullptr;
            dfsStack.emplace_back(rootNode);

            while (dfsStack.size() > 0) {
                FTMemCell *curNode = dfsStack.back();
                dfsStack.pop_back();

                if (curNode->allocationSize > 0) {
                    //todo: inefficient saving format
                    //This stack trace has at least one driverMemRecord allocation, print
                    this->printCallStack(curNode, flameGraphOutput);
                }
                //Insert all children into the dfs stack
                for (auto &valuePair: curNode->children) {
                    valuePair.second->outputParent = curNode; //Set parent at output time
                    dfsStack.emplace_back(valuePair.second);
                }
            }

            summaryOutput << "Pytorch State:"
                          << hookInstallerInstance->pytorchModuleInfoMap[snapshotInfo.pyTorchModuleStack.pyTorchModuleId].moduleName
                          << '(' << toString(snapshotInfo.pyTorchModuleStack.pyTorchModuleExecutionState) << ')'
                          << std::endl;


            summaryOutput << "Flame Graph Summary:" << std::endl;
            for (ssize_t attributionId = 0; attributionId < aggregationByPyModule->size(); ++attributionId) {
                ssize_t memoryInBytes = (*aggregationByPyModule)[attributionId];
                if (memoryInBytes > 0) {
                    summaryOutput << "\t" << pyModuleSummaryAttributionArray[attributionId].friendlyName.c_str() << ": "
                                  << format_size(memoryInBytes) << std::endl;
                }
            }
            if (donotcareModules > 0) {
                summaryOutput << "\t" << "Others: " << format_size(donotcareModules) << std::endl;
            }
            summaryOutput << std::endl;
        }

        ~FlameTree<FTMemCell>() {
            //todo: Free driverMemRecord for all snapshots at the end

        };

        /**
         * Print summary
        */
    protected:
        std::vector<FlameGraphSnapshot> snapShotArray;
        ssize_t curSnapShotVersion = 0; // The index to snapShotArray, which marks the root nodes for flamegraph snaoshpts
        ObjectPoolHeap<FTMemCell> nodePool;
        ssize_t donotcareModules = 0;

        void printCallStack(FTMemCell *curCallStackLevel, FILE *flameGraphOutput) {
            HookInstaller *instance = HookInstaller::getInstance();

            //todo: inefficient DFS due to data conversion
            std::vector<FTMemCell *> memCellStackQueue;

            FTMemCell *curNode = curCallStackLevel;

            //todo: Handle the root node
            while (curNode->outputParent != nullptr) {
                PyModuleInfo &curModuleInfo = instance->pyModuleInfoMap.operator[](curNode->pyModuleId);
                if (curModuleInfo.moduleType == PyModuleType::USER_CARE) {
                    (*aggregationByPyModule)[curModuleInfo.pyPackageSummaryAttributionId] += curNode->allocationSize;
                } else {
                    donotcareModules += curNode->allocationSize;
                }

                memCellStackQueue.emplace_back(curNode);
                curNode = curNode->outputParent;
            }


            ssize_t stackSize = memCellStackQueue.size();
            for (ssize_t i = stackSize - 1; i >= 0; --i) {
                fprintf(flameGraphOutput, "%s; ",
                        instance->pyModuleInfoMap[memCellStackQueue[i]->pyModuleId].moduleName.c_str());
            }
            fprintf(flameGraphOutput, "%zd\n", curCallStackLevel->allocationSize);
        }

    };
}
namespace mlinsight::FlameGraph {
    /**
    * The FrameworkTensorMixin necessary for all classes in mlinsight::MemLeak::InternalFrag
    */
    class TensorMixin {
    public:
        /* A corresponding flame graph node. This data is mainly used at the deallocation time to reduce the overhead of walking the stack to find the corresponding node.
        This pointer does not necessarily point to the latest flame graph snapshot, that is why it should not be modifed by classes other than FlameGraph. */
        FTMemCell const *flameGraphCell = nullptr;

        TensorMixin(ssize_t size, void *ptr){
            //Do not need to do anything here.
        }
    };
}
namespace mlinsight{
    template<typename CTENSOR_TYPE>
    class FlameGraphAnalyser:public SimpleCallback<CTENSOR_TYPE>{
    public:
        FlameTree<FTMemCell> pyMemoryFlameTreeByPyPackage;
    public:

        /**
        * [Interface]
        * Invoked after the allocator allocates memory.  Insert a new Tensor into mapAliveObjs.
        * For each allocation, the sequence is [onPostAlloc(...... AllocationType::Driver] -> [onPostAlloc(...... AllocationType::Framework]
        * @param size The size of the allocation
        * @param ptr Memory pointer
        * @param type Indicate whether this is a driver allocation or framework allocation.
        */
        void onPostAlloc(ssize_t size, void *ptr, CTENSOR_TYPE* newTensor) {
            if(newTensor){
                auto* flamegraphTensor = static_cast<mlinsight::FlameGraph::TensorMixin*>(newTensor);
                //Save the shortcut for
                //flamegraphTensor->flameGraphCell=pyMemoryFlameTreeByPyPackage.addData(newTensor->callstack,newTensor->size);
            }
        }

        /**
        * [Interface]
        * Invoked before the allocator frees memory. Remove a new Tensor from mapAliveObjs.
        * For each allocation, the sequence is [onPostAlloc(...... AllocationType::Framework] -> [onPostAlloc(...... AllocationType::Driver]
        * @param ptr Memory pointer
        * @param type Indicate whether this is a driver allocation or framework allocation.
        */
        void onPreFree(void *ptr,CTENSOR_TYPE* justFreedTensor) {
            //todo:Finish here
            if(justFreedTensor){
                //pyMemoryFlameTreeByPyPackage.rmData(justFreedTensor->flameGraphCell, justFreedTensor->callstack,justFreedTensor->size);
            }
        }


    };



}
#endif //MLINSIGHT_FLAMEGRAPH_H
