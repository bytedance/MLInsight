#ifndef MLINSIGHT_LINKEDLIST_H
#define MLINSIGHT_LINKEDLIST_H

/*
 * @file   LinkedList.h
 * @author
 *         Steven (Jiaxun) Tang<jtang@umass.edu>
 *         The original design is from:
 *         Tongping Liu <http://www.cs.umass.edu/~tonyliu>
 *         kulesh [squiggly] isis.poly.edu
 */

#include <utility>
#include <cassert>
#include "common/MemoryHeap.h"


namespace mlinsight {
    template<typename VALUE_TYPE, template<typename> class HEAP_TYPE>
    class LinkedList;

    template<typename VALUE_TYPE, template<typename> class HEAP_TYPE>
    class ListIterator;

    /**
     * Node class in linked list.
     * @tparam VALUE_TYPE The value type stored in this node.
     * @tparam HEAP_TYPE Heap type used to allocate this node, mainly used in LinkedList.
     */
    template<typename VALUE_TYPE, template<typename> class HEAP_TYPE>
    class ListEntry {
    public:
        friend class LinkedList<VALUE_TYPE, HEAP_TYPE>;

        friend class ListIterator<VALUE_TYPE, HEAP_TYPE>;

        //Previous node.
        ListEntry *prev;
        //Next node.
        ListEntry *next;
        //The value of this list entry not constructed at the creation time of ListEntry. See LazyConstructValue for details.
        LazyConstructValue<VALUE_TYPE> value;

        /**
         * Construct ListEntry.
         * @param prev Previous node pointer.
         * @param next Next node pointer.
         */
        ListEntry(ListEntry *prev = nullptr, ListEntry *next = nullptr) : prev(prev), next(next) {

        }

        bool operator==(const ListEntry &rho) const {
            //Delegate comparison to this->value::operator==.
            return this->value == rho.value;
        }

        bool operator!=(const ListEntry &rho) const {
            return !operator==();
        }

        /**
         * Get constructed driverMemRecord of this->value. If this->value is not constructed, then this function will abort the
         * program because it is the developer's fault.
         * @return Reference to VALUE_TYPE.
         */
        inline VALUE_TYPE &getValue() {
            return this->value.getConstructedValue();
        }

        inline ListEntry *getPrev() {
            return this->prev;
        }

        inline ListEntry *getNext() {
            return this->next;
        }
    };

    /**
     * Iterator for list.
     * @tparam VALUE_TYPE The value type stored in this node.
     * @tparam HEAP_TYPE Heap type used to allocate this node, used mainly in LinkedList.
     */
    template<typename VALUE_TYPE, template<typename> class HEAP_TYPE>
    class ListIterator {
    protected:
        using Linkedist_t = LinkedList<VALUE_TYPE, HEAP_TYPE>;
        using ListEntry_t = ListEntry<VALUE_TYPE, HEAP_TYPE>;

        friend class LinkedList<VALUE_TYPE, HEAP_TYPE>;

        // The owner of this iterator
        Linkedist_t *list = nullptr;
        // Current node in the list.
        ListEntry_t *curNode = nullptr;

        explicit ListIterator(Linkedist_t *list) : list(list) {
        }

        /*
         * Go to next node with iterator++.
         * If there is no next node, then this function will abort the program.
         */
        inline void operator++() {
            assert(this->curNode != this->list->tail);
            this->curNode = this->curNode->next;
        }

        /*
         * Go to next node with ++iterator.
         * If there is no next node, then this function will abort the program.
         */
        inline void operator++(int) {
            operator++();
        }

        /**
         * Go to previous node with iterator--.
         * If there is no previous node, then this function will abort the program.
         */
        inline void operator--() {
            assert(this->curNode != this->list->head);
            this->curNode = this->curNode->prev;
        }

        /**
         * Go to previous node with --iterator.
         * If there is no previous node, then this function will abort the program.
         */
        inline void operator--(int) {
            operator--();
        }

        bool operator==(const ListIterator &rho) const {
            //First check if list object is the same, then check whether the list entry object is the same.
            //See ListEntry::operator=.
            return this->list == rho.list && this->curNode == rho.curNode;
        }

        bool operator!=(const ListIterator &rho) const {
            return !operator==(rho);
        }

        /**
         * Get the value of objects with *iterator. The value of Listnode is lazily constructed.
         * If value is not constructed, then the program will abort because it is the developer's fault.
         * @return Reference to the value in current node.
         */
        VALUE_TYPE &operator*() {
            assert(this->curNode != nullptr && this->curNode != this->list->tail);
            return reinterpret_cast<VALUE_TYPE &>(this->curNode->getValue());
        }
    };


    /**
     * Double linked list
     * @tparam VALUE_TYPE Value type of all list nodes.
     */
    template<typename VALUE_TYPE, template<typename> class HEAP_TYPE=PassThroughMemoryHeap>
    class LinkedList {
    protected:
        using ListEntry_t = ListEntry<VALUE_TYPE, HEAP_TYPE>;
        using ListIterator_t = ListIterator<VALUE_TYPE, HEAP_TYPE>;

        friend class ListIterator<VALUE_TYPE, HEAP_TYPE>;

        //Head dummy node, do not hold value.
        ListEntry_t *head;
        //Tail dummy node, do not hold value.
        ListEntry_t *tail;

        //Memory heap used to allocate listentry.
        HEAP_TYPE<ListEntry_t> listElementHeap;

        //beginIterator returned by this->begin(). It points to this->head->next.
        ListIterator_t beginIter;
        //endIterator returned by this->end(). It points to the this->tail.
        ListIterator_t endIter;
        //reverseBeginIterator returned by this->rbegin(). It points to this->tail->prev.
        ListIterator_t rbeginIter;
        //reverseEndIterator returned by this->rbegin(). It points to this->head.
        ListIterator_t rendIter;

        //The amount of all nodes in this list
        ssize_t size = 0;
    public:
        LinkedList() : beginIter(this), endIter(this), rbeginIter(this), rendIter(this),
                       listElementHeap() {
            //Allocate head and tail node from heap.
            head = listElementHeap.generalStats();
            tail = listElementHeap.generalStats();
            //Construct head and tail node. We do not use new here to keep consistency with other nodes.
            new(head) ListEntry_t(nullptr, tail);
            new(tail) ListEntry_t(head, nullptr);
            //Set current node in iterator
            beginIter.curNode = head->next;
            endIter.curNode = tail;
            rbeginIter.curNode = tail->next;
            rendIter.curNode = head;
        };

        /**
         * Copy and swap idiom: Copy constructor.
         * Construct self as a temporary object based on rho.
         */
        LinkedList(const LinkedList &rho) : LinkedList() {
            //We do not actually modify value, but need to invoke non-const code.
            auto &rhoNonConst = const_cast<LinkedList &>(rho);

            ListEntry_t *rhoHead = rhoNonConst.getHead();
            ListEntry_t *rhoTail = rhoNonConst.getTail();

            //Insert all nodes in rho into this list
            ListEntry_t *curRhoEntry = rhoHead->getNext();
            while (curRhoEntry != rhoTail) {
                insertBack(curRhoEntry->getValue());
                curRhoEntry = curRhoEntry->getNext();
            }
        }

        /**
         * Copy and swap idiom: Copy assignment.
         */
        LinkedList &operator=(const LinkedList &rho) {
            if (this != &rho) {
                LinkedList tempObject(rho);
                //Copy and swap idiom. Create temporary object using copy constructor.
                swap(*this, tempObject);
            }
            return *this;
        }

        /**
         * Copy and swap idiom: Move constructor.
         */
        LinkedList(LinkedList &&rho) noexcept: head(rho.head), tail(rho.tail), listElementHeap(rho.listElementHeap),
                                               beginIter(rho.beginIter), endIter(rho.endIter),
                                               rbeginIter(rho.rbeginIter),
                                               rendIter(rho.rendIter) {
            //Prevent rho destructor from freeing driverMemRecord head and tail driverMemRecord
            rho.head = nullptr;
            rho.tail = nullptr;
            //Modify iterator list pointer to this.
            beginIter.list = this;
            endIter.list = this;
            rbeginIter.list = this;
            rendIter.list = this;
        }

        /**
         * Copy and swap idiom: Move assignment.
         */
        LinkedList &operator=(LinkedList &&rho) {
            swap(*this, rho);
            return *this;
        }

        /**
         * Copy and swap idiom: swap function.
         */
        friend void swap(LinkedList &lho, LinkedList &rho) noexcept {
            using std::swap; //Make swap fallback to std::swap.
            //Swap will not allocateArray extra driverMemRecord, which is ensured by the compiler os there will be no exception.
            swap(lho.head, rho.head);
            swap(lho.tail, rho.tail);
            swap(lho.listElementHeap, rho.listElementHeap);
            swap(lho.beginIter, rho.beginIter);
            swap(lho.endIter, rho.endIter);
            swap(lho.rbeginIter, rho.rbeginIter);
            swap(lho.rendIter, rho.rendIter);
            swap(lho.size, rho.size);
        }

        virtual ~LinkedList() {
            if (head != nullptr) {
                clear();
                tail->~ListEntry<VALUE_TYPE, HEAP_TYPE>();
                listElementHeap.dealloc(tail);
                head->~ListEntry<VALUE_TYPE, HEAP_TYPE>();
                listElementHeap.dealloc(head);
            }
        }

        /**
         * Free all nodes in this list
         */
        void clear() {
            ListEntry_t *curNode = head->next;
            while (curNode != tail) {
                ListEntry_t *nextNode = curNode->next;
                //Manually call destructor because we used placement new.
                curNode->~ListEntry_t();
                //Free driverMemRecord of ListEntry, listElementHeap will not free ListEntry.
                listElementHeap.dealloc(curNode);
                curNode = nextNode;
            }
            head->next = tail;
            size = 0;
        }

        bool isEmpty() {
            // Head and tail do not have value. If empty, then head->next == tail.
            return head->next == tail;
        }

        /**
         * Insert new node after specified node. Construct the node with arguments passed in "args".
         * If the user do not want to construct VALUE_TYPE, please check insertAfterLazyConstruct
         * @tparam Args Perfect forwarding arguments to the constructor of VALUE_TYPE.
         * @param node The new node will be inserted after this node.
         * @param args Perfect forwarding arguments to the constructor of VALUE_TYPE.
         * @return A reference to constructed value stored in newly created list entry.
         */
        template<typename... Args>
        VALUE_TYPE &insertAfter(ListEntry_t *node, Args &&... args) {
            LazyConstructValue<VALUE_TYPE> &rawMemory = insertAfterLazyConstruct(node);
            //Forward all arguments to VALUE_TYPE
            rawMemory.constructValue(std::forward<Args>(args)...);
            return rawMemory.getConstructedValue();
        }

        /**
         * Insert new node at the end of this list. Construct the node with arguments passed in "args".
         * If the user do not want to construct VALUE_TYPE, please check insertAfterLazyConstruct
         * @tparam Args Perfect forwarding arguments to the constructor of VALUE_TYPE.
         * @param args Perfect forwarding arguments to the constructor of VALUE_TYPE.
         * @return A reference to constructed value stored in newly created list entry.
         */
        template<typename... Args>
        VALUE_TYPE &insertBack(Args &&... args) {
            return insertAfter(this->tail->prev, std::forward<Args>(args)...);
        }

        /**
         * Insert new node after specified node. Do not construct the node and only allocates driverMemRecord.
         * The user must construct VALUE_TYPE with "placement new" before first accessing driverMemRecord.
         * The user do NOT have to destruct this because it will be freed with ListEntry.
         * @param node The new node will be inserted after this node.
         */
        LazyConstructValue<VALUE_TYPE> &insertAfterLazyConstruct(ListEntry_t *node) {
            //Cannot insert value in a null node
            assert(node != nullptr);
            //Cannot insert after tail
            assert(node != tail);

            //Allocate driverMemRecord for a new entry
            ListEntry_t *newEntry = listElementHeap.generalStats();
            //Construct ListEntry
            new(newEntry) ListEntry_t();
            //Insert this entry to list
            newEntry->next = node->next;
            newEntry->prev = node;
            node->next = newEntry;
            if (newEntry->next)
                newEntry->next->prev = newEntry;
            ++size;
            return newEntry->value;
        }

        /**
         * Erase a node and destruct node value.
         * If a node is not constructed, this function will abort program because it's developer's fault.
         * @param node The node to remove.
         */
        void erase(ListEntry_t *node) {
            assert(node != nullptr);
            assert(node != head);
            assert(node != tail);

            //Remove node
            ListEntry_t *nodePrev = node->prev;
            ListEntry_t *nodeNext = node->next;
            nodePrev->next = nodeNext;
            nodeNext->prev = nodePrev;
            //Destruct node
            node->~ListEntry_t();
            //Free driverMemRecord
            listElementHeap.dealloc(node);
            node = nullptr;
            --size;
        }

        inline ListEntry_t *getHead() {
            return head;
        }

        inline ListEntry_t *getTail() {
            return tail;
        }

        inline const ssize_t &getSize() const {
            return size;
        }

        /**
         * Return begin iterator. *begin points to list->head->next.
         * @return Const begin iterator, user must create copy to manipulate iterator.
         */
        const ListIterator_t &begin() {
            beginIter.curNode = head->next;
            return beginIter;
        }

        /**
         * Return end iterator. *end points to list->tail.
         * @return Const end iterator, user must create copy to manipulate iterator.
         */
        const ListIterator_t &end() {
            return endIter;
        }

        /**
         * Return reverse begin iterator. *rbegin points to list->tail->prev.
         * @return Const rbegin iterator, user must create copy to manipulate iterator.
         */
        const ListIterator_t &rbegin() {
            rbeginIter.curNode = tail->prev;
            return rbeginIter;
        }

        /**
         * Return reverse end iterator. *rend points to list->head.
         * @return Const rend iterator, user must create copy to manipulate iterator.
         */
        const ListIterator_t &rend() {
            return rendIter;
        }

    };


}
#endif
