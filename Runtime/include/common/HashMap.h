#ifndef MLINSIGHT_HASHMAP_H
#define MLINSIGHT_HASHMAP_H

/*
 * @file   LinkedList.h
 * @author
 *         Steven (Jiaxun) Tang<jtang@umass.edu>
 */
#include <unistd.h>
#include <pthread.h>
#include <new>
#include <cassert>
#include <cstdlib>
#include "common/Logging.h"
#include "common/LinkedList.h"
#include "common/LazyValueType.h"
#include "common/HashAndCompareFunctions.h"

namespace mlinsight {
    template<class KEY_TYPE, class VALUE_TYPE,template<typename> class HEAP_TYPE>
    class HashMap;

    /**
     * A key and value pair.
     * @tparam KEY_TYPE Type of key.
     * @tparam VALUE_TYPE Type of value.
     * @tparam HEAP_TYPE Type of memory allocator for HashBucket.
     */
    template<typename KEY_TYPE, typename VALUE_TYPE,template<typename> class HEAP_TYPE>
    class HashEntry {
    protected:
        //Allow Hashmap to access hash entry.
        friend class HashMap<KEY_TYPE, VALUE_TYPE,HEAP_TYPE>;
        //Key stored in this hash entry.
        KEY_TYPE key;
        //Value stored in this hash entry. This value is constructed lazily, check LazyConstructValue.
        LazyConstructValue<VALUE_TYPE> value;

    public:

         /**
          * Construct HashEntry.
          * @param key The key of this hash entry.
          */
        HashEntry(KEY_TYPE &key) : key(key) {
        }

        /**
          * Construct HashEntry.
          * @param key The key of this hash entry. (rvalue)
          */
        HashEntry(KEY_TYPE &&key) : key(key) {
        }

        inline const KEY_TYPE &getKey() const {
            return key;
        }

        /**
         * Get the value stored in this hash entry. The value must have been constructed before invoking this function,
         * otherwise this function will abort the program because it is the developer's fault.
         */
        inline VALUE_TYPE& getValue() {
            return value.getConstructedValue();
        }


        bool operator==(const HashEntry &rho) const {
            //Only compare key because this in hashmap. Delegate comparison to KEY_TYPE::operator==.
            return key == rho.key;
        }

        bool operator!=(const HashEntry &rho) const {
            return !operator==();
        }

    };

    /**
     * Iterator for hashmap.
     * @tparam KEY_TYPE Type of key.
     * @tparam VALUE_TYPE Type of value.
     * @tparam HEAP_TYPE Type of memory allocator for HashBucket.
     */
    template<class KEY_TYPE, class VALUE_TYPE,template<typename> class HEAP_TYPE>
    class HashMapIterator{
    protected:
        using HashMap_t = HashMap<KEY_TYPE, VALUE_TYPE,HEAP_TYPE>;
        using HashBucket_t = LinkedList<HashEntry<KEY_TYPE, VALUE_TYPE,HEAP_TYPE>,HEAP_TYPE>;
        using HashEntry_t = HashEntry<KEY_TYPE, VALUE_TYPE,HEAP_TYPE>;
        using BucketEntry_t = ListEntry<HashEntry_t,HEAP_TYPE>;
        //Allow hashmap to access iterator.
        friend class HashMap<KEY_TYPE, VALUE_TYPE,HEAP_TYPE>;

        // The owner of this iterator.
        HashMap_t *hashMap;
        // The current bucket id this iterator points to.
        ssize_t curBucketId;
        // The current bucket entry this iterator points to.
        BucketEntry_t *curBucketEntry;
    public:
        explicit HashMapIterator(HashMap_t *hashMap):curBucketId(0),hashMap(hashMap),curBucketEntry(nullptr) {

        }

        /**
         * Go to next element, use iterator++.
         * If there are no next element, then this function will abort the program because it is the developer's fault.
         */
        void operator++()  {
            bool hasNext = getNextEntry(curBucketId, curBucketEntry);
            assert(hasNext);
        }

        /**
         * Go to next element, use ++iteratorx.
         * If there are no next element, then this function will abort the program because it is the developer's fault.
         */
        void operator++(int)  {
            operator++();
        }


        bool operator==(const HashMapIterator &rho) const  {
            //Equal only when the bucket is the same.
            return curBucketEntry == rho.curBucketEntry && curBucketId == rho.curBucketId &&
                   hashMap == rho.hashMap;
        }

        bool operator!=(const HashMapIterator &rho) const {
            return !operator==(rho);
        }

        /**
         * *iterator to get the constructed value.
         * If the value is not constructed, this function will abort the program because it is the developer's fault.
         * @return Constructed value.
         */
        VALUE_TYPE &operator*()  {
            assert(curBucketEntry != hashMap->buckets[curBucketId].getTail());
            return curBucketEntry->getValue().getValue();
        }

        const KEY_TYPE &getKey() {
            //The getValue returns HashEntry, the getKey returns value HashEntry::key.
            return curBucketEntry->getValue().getKey();
        }

        VALUE_TYPE &getValue() {
            return operator*();
        }

    protected:
        /**
         * Move to the next entry.
         * @param retBucketId The returned bucket id of next entry.
         * @param retEntry The returned bucket entry.
         * @return This this invocation successfully move to the next entry.
         */
        inline bool getNextEntry(ssize_t& retBucketId, BucketEntry_t *&retEntry) {
            bool hasNextEntry = false;
            retEntry = curBucketEntry;
            HashBucket_t * bucket=&(hashMap->buckets[retBucketId]);
            while (true) {
                if (retEntry->getNext() != bucket->getTail()) {
                    //Not tail, go to next element.
                    retEntry = retEntry->getNext();
                    hasNextEntry = true;
                    break;
                } else if (retBucketId == hashMap->bucketNum - 1) {
                    //Last bucket move to next retEntry only
                    retEntry = retEntry->getNext();
                    hasNextEntry = true;
                    break;
                } else {
                    //Not last bucket but is tail node, move to the next bucket
                    bucket = &hashMap->buckets[++retBucketId];
                    retEntry = bucket->getHead();
                }
            }
            return hasNextEntry;
        }

    };

    template<typename KEY_TYPE, typename VALUE_TYPE,template<typename> class HEAP_TYPE=PassThroughMemoryHeap>
    class HashMap {
    protected:
        using HashEntry_t = HashEntry<KEY_TYPE, VALUE_TYPE,HEAP_TYPE>;
        using BucketEntry_t = ListEntry<HashEntry_t,HEAP_TYPE>;
        using HashBucket_t = LinkedList<HashEntry_t,HEAP_TYPE>;
        using HashMapIterator_t = HashMapIterator<KEY_TYPE, VALUE_TYPE,HEAP_TYPE>;
        friend class HashMapIterator<KEY_TYPE, VALUE_TYPE,HEAP_TYPE>;

        typedef bool (*Comparator_t)(const KEY_TYPE &src, const KEY_TYPE &dst);

        typedef size_t (*HashFunc_t)(const KEY_TYPE &key);

        HashBucket_t *buckets = nullptr; //There are bucketNum+1 nodes. The last node indicates the end of bucket.
        ssize_t bucketNum = -1;     // How many buckets in total
        HashFunc_t hfunc = nullptr;
        Comparator_t kcmp = nullptr;
        ssize_t elementSize=0; //How many elements are there in this hashmap
        HashMapIterator_t beginIter;
        HashMapIterator_t endIter;
    public:

        HashMap(const ssize_t bucketNum = 16, HashFunc_t hfunc = hash<KEY_TYPE>,Comparator_t kcmp = compare<KEY_TYPE>):
                            hfunc(hfunc),kcmp(kcmp),buckets(nullptr),bucketNum(bucketNum),beginIter(this),endIter(this){
            assert(bucketNum >= 1);
            assert(hfunc != nullptr && kcmp != nullptr);
            // Initialize all of these _entries.
            this->buckets = reinterpret_cast<HashBucket_t*>(malloc(sizeof(HashBucket_t)*bucketNum));
            if(!this->buckets){
                fatalError("Cannot allocateArray memory for hash buckets.");
            }
            for (int i = 0; i < bucketNum; ++i) {
                //Do not use "new HashBucket_t[]" to allocate memory because we may need to use custom constructor parameter
                new (this->buckets+i) HashBucket_t();
            }
        }

        /**
         * Copy and swap idiom: Copy constructor.
         */
        HashMap(const HashMap& rho):bucketNum(rho.bucketNum),hfunc(rho.hfunc),kcmp(rho.kcmp),elementSize(rho.elementSize),
                                                                                            beginIter(this),endIter(this){
            this->buckets = reinterpret_cast<HashBucket_t*>(malloc(sizeof(HashBucket_t)*bucketNum));
            if(!this->buckets){
                fatalError("Failed to allocateArray memory for HashBuckets.")
            }
            for (int i = 0; i < bucketNum; ++i) {
                //Delegate copy constructor to HashBucket_t's copy constructor
                new (this->buckets+i) HashBucket_t(rho.buckets[i]);
            }

        }

        /**
         * Copy and swap idiom: Move constructor.
         */
        HashMap(HashMap&& rho):bucketNum(rho.bucketNum),hfunc(rho.hfunc),kcmp(rho.kcmp),elementSize(rho.elementSize),
                                        beginIter(this),endIter(this){
            //Here, we do not allocateArray buckets, but will directly use buckets from rho.
            HashBucket_t* oldMemory=this->buckets;
            this->buckets=rho.buckets;
            rho.buckets=nullptr; //Prevent rho from freeing the memory.
            free(oldMemory); //Free previously allocated memory
        }

        /**
         * Copy and swap idiom: Copy assignment.
         */
        HashMap& operator=(const HashMap& rho){
            if(this!=&rho){
                HashMap tempObj(rho);
                swap(*this,tempObj);
            }
            return *this;
        }

        /**
         * Copy and swap idiom: Move assignment.
         */
        HashMap& operator=(HashMap&& rho){
            swap(*this,rho);
            return *this;
        }

        /**
         * Copy and swap idiom: swap function.
         */
        friend void swap(HashMap& lho,HashMap& rho) noexcept{
            using std::swap;
            //Swap will not allocateArray extra memory, which is ensured by the compiler.
            //The memory allocated in this class will be freed with tempObj.
            swap(lho.buckets,rho.buckets);
            swap(lho.bucketNum,rho.bucketNum);
            swap(lho.hfunc,rho.hfunc);
            swap(lho.kcmp,rho.kcmp);
            swap(lho.elementSize,rho.elementSize);
            swap(lho.beginIter,rho.beginIter);
            swap(lho.endIter,rho.endIter);
        }

        virtual ~HashMap() {
            if(buckets){
                //Free buckets
                for (int i = 0; i < bucketNum; ++i) {
                    //Free HashBucket value
                    buckets[i].~HashBucket_t();
                }
                free(buckets);
            }
        }

        inline ssize_t hashIndex(const KEY_TYPE &key) {
            size_t hkey = hfunc(key);
            return hkey % bucketNum;
        }


        /**
         * Find an object from map by key.
         * If the user needs to first check whether key exists and then insert. Please use findAndInsert API instead
         * rather then call "find" first and then call "insert" because doing so will perform find twice.
         * If this object exists, return HashEntry object.
         * If this object does not exist, return null.
         */
        VALUE_TYPE *find(const KEY_TYPE &key) {
            ssize_t hindex = hashIndex(key);
            HashEntry_t *hashEntry;
            HashBucket_t *bucket = getHashBucket(hindex);
            bool found = getHashEntry(key, bucket, hashEntry);

            return found ? &(hashEntry->getValue()) : nullptr;
        }

        /**
         * Insert an element into hashmap with key. If key already exists, then the value will be replaced.
         * If the user do not wish to insert non-exist element automatically, please use "find" API.
         */
        template<typename KEY_TYPE_=KEY_TYPE,typename ...Args>
        inline VALUE_TYPE& insert(KEY_TYPE_ &&key,Args&&... args) {
            bool retFound;
            return findAndInsert(key,true,retFound,std::forward<Args>(args)...);
        }

        /**
         * Insert an element into hashmap with key. If key already exists, then the value will be replaced.
         * Compared to "insert" API, this API will further return whether the key exists before insertion.
         * This function is suitable for find first if not found perform insertion situation because there is no need to perform find twice.
         * @tparam KEY_TYPE_ Type of key for perfect forwarding.
         * @tparam Args Argument pack for perfect forwarding.
         * @param key Key
         * @param replace If true, element will be replace if the key exists
         * @return retFound Whether key exists in hashmap before invoking this functionl.
         * @param args Argument pack for perfect forwarding.
         * @return Reference to the value of hashmap element.
         */
        template<typename KEY_TYPE_=KEY_TYPE,typename ...Args>
        inline VALUE_TYPE& findAndInsert(KEY_TYPE_ &&key,bool replace,bool& retFound,Args&&... args) {
            LazyConstructValue<VALUE_TYPE>& rawValue= insertLazyConstruct(key, retFound);
            if(replace){
                //Should replace
                //If value exists, destruct and replace. If value not exists, construct.
                if(retFound){
                    rawValue.destructValue();
                }
                rawValue.constructValue(std::forward<Args>(args)...);
            }else if(!retFound){
                //Should not replace and key is not found (new element allocated). Construct.
                rawValue.constructValue(std::forward<Args>(args)...);
            }else{
                //Should not replace but key is found, return directly.
            }


            return rawValue.getConstructedValue();
        }

        /**
         * Free an entry of hashmap and destruct it.
         * @param key
         * @param mustExist
         */
        void erase(const KEY_TYPE &key, bool mustExist = false) {
            ssize_t hindex = hashIndex(key);

            auto *curBucket = getHashBucket(hindex);

            BucketEntry_t *entry;
            bool isFound = getBucketEntry(key, curBucket, entry);

            if (mustExist)
                assert(isFound);
            if (isFound) {
                //Call the destructor of VALUE
                curBucket->erase(entry);
                elementSize-=1;
            }
        }
        /**
         * Remove all elements in hashmap
         */
        void clear(){
            for(int i=0;i<bucketNum;++i){
                buckets[i].clear();
            }
            elementSize=0;
        }

        /**
         * Acquire the first entry of the hash table.
         * If there is no entry, then begin()==end()==head of the first bucket
         * @return
         */
        const HashMapIterator_t &begin() {
            beginIter.curBucketId = 0;
            beginIter.curBucketEntry = this->buckets[beginIter.curBucketId].getHead();
            ++beginIter;
            return beginIter;
        }

        const HashMapIterator_t &end() {
            endIter.curBucketId = bucketNum - 1;
            endIter.curBucketEntry = this->buckets[endIter.curBucketId].getTail();
            return endIter;
        }


        const ssize_t& getSize() const {
            return elementSize;
        }


    protected:

        inline HashBucket_t *getHashBucket(const size_t &hindex) {
            assert(hindex < bucketNum);
            return &buckets[hindex];
        }

        /**
         * Return the first found entry or the last element of entry list if not found
         * @return :Found entry or not
         */
        inline bool
        getHashEntry(const KEY_TYPE &key, HashBucket_t *bucket, HashEntry_t *&entry) {
            BucketEntry_t *hashEntry;
            bool found = getBucketEntry(key, bucket, hashEntry);
            if (found)
                entry = &hashEntry->getValue();
            return found;
        }


        inline bool
        getBucketEntry(const KEY_TYPE &key, HashBucket_t *bucket, BucketEntry_t *&entry) {
            assert(bucket != nullptr);
            bool found = false;
            if (bucket != nullptr) {
                //Start with the first node
                auto *listEntry = bucket->getHead()->getNext();

                while (listEntry != bucket->getTail()) {
                    if (kcmp(listEntry->getValue().getKey(), key)) {
                        found = true;
                        break;
                    }
                    listEntry = listEntry->getNext();
                }

                if (found)
                    entry = listEntry;

            }
            return found;
        }

        /**
         * Insert an element into hashmap with key. If key already exists, then the value will be replaced.
         * Compared to "Insert" and "findAndInsert", this function will only allocate memory but willnot construct object.
         * The user needs to construct object using placement-new at appropriate times.
         * The user do not need to free this object because it will be freed by HashMap.
         */
        template<typename KEY_TYPE_=KEY_TYPE>
        LazyConstructValue<VALUE_TYPE>& insertLazyConstruct(KEY_TYPE_ &&key, bool& retFound) {
            ssize_t hindex = hashIndex(key);
            HashBucket_t *bucket = getHashBucket(hindex);
            HashEntry_t *hashEntry;
            retFound=getHashEntry(key, bucket, hashEntry);
            if (!retFound) {
                //Did not find this hashEntry (collision), so we have to insert current key into hash bucket
                //Construct key but do not construct value. Value construction is delayed to user.
                hashEntry=&(bucket->insertAfter(bucket->getHead(),key));
            }

            elementSize+=1;
            //Return value part to user
            return hashEntry->value;
        }

    };


}
#endif
