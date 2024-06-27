#ifndef MLINSIGHT_CALLSTACK_H
#define MLINSIGHT_CALLSTACK_H

#include <cstring>
#include <cassert>
#include <iostream>
#include <fstream>
#include <execinfo.h>
#include <vector>
#include "common/Logging.h"
#include "common/MemoryHeap.h"
#include <set>
#include <unordered_map>
#include "trace/type/RecordingDataStructure.h"

/**
 * author: Steven Tang <stevne.tang@bytedance.com>
 */

namespace mlinsight
{
    constexpr int CPP_CALL_STACK_LEVEL = 20;
    constexpr int PYTHON_CALL_STACK_LEVEL = 20;

    namespace callstack
    {

        /**
         * This class is used as an interface. Please do not subclass upcast to this type.
         * To save overhead, we do not use virtual method. Developers should
         * @tparam ARRAY_ELEMENT_POINTER_TYPE
         * @tparam MAX_LEVEL (This is only seen as a suggested value. In practice, subclasses may ignore it if necessary.)
         */
        template <typename CALLSTACK_KEY, int MAX_LEVEL>
        class CallStack
        {
        public:
            // A pointer to an void* array storing callstack
            std::vector<CALLSTACK_KEY> array;
            ssize_t callstackID=-1;
            bool isNewCallStackId=false; // A flag indicating whether the current callstack first sees the callstack.


        public:
            CallStack() : array(MAX_LEVEL)
            {
            }

            bool operator==(const CallStack &rho) const
            {
                bool isEqual = this->array.size() == rho.array.size();
                if (isEqual)
                {
                    for (int i = 0; i < rho.array.size(); ++i)
                    {
                        if (!(this->array[i] == rho.array[i]))
                        {
                            isEqual = false;
                            break;
                        }
                    }
                }
                return isEqual;
            }

            /**
             * Take a snapshot of the current callstack and store in this object.
             */
            void snapshot()
            {
                fatalError("Abstract interface. Please override this function in subclass");
            }
            /**
             * Parse complete callstack information
             */
            void lazyParse()
            {
                fatalError("Abstract interface. Please override this function in subclass");
            }

            /**
             * Output the callstack information into std::ostream
             */
            void print(std::ostream& os)
            {
                fatalError("Abstract interface. Please override this function in subclass");
            }

            std::string toString(){
                fatalError("Abstract interface. Please override this function in subclass");
                return "";
            }

            size_t hash() const{
                fatalError("Abstract interface. Please override this function in subclass");
            }
        };

        class FrameExtra;
        /**
         * FrameKey calsses represents the minimum information needed for a callstack line.
         *
         * Collecting the complete stack information may be slow, but information related to the stack trace must be collected at invocation time.
         * So only a minimum possible set of information should be collected at invocation time. If this stack information is indeed used later, more complete inforamtion should be parsed lazily. "EXTRAINFO_T extra" serves this purpose.
         * 
         * The default frameKey uses a uint64_t pointer as the line key. 
         * 
         * Note that two callstacks lines are the same if and only if their keys are the same. The "extra" field is linked with the key but is not involved in the equality comparison.
         * 
         * FrameKey is a default storage structure in callstack, so it should be kept minimum size.
         */
        template<typename Key_T, typename EXTRAINFO_T>
        class FrameKey
        {
        public:
            Key_T key;
            EXTRAINFO_T* extra;
        public:

            FrameKey()=default;

            FrameKey(Key_T key, EXTRAINFO_T* extra):key(key),extra(extra){

            }

            /**
             * Compare callstack keys
             */
            bool operator==(const FrameKey &rho)
            {
                fatalError("Please use subclass");
                return false;
            }

            /**
             * Returns the hash of the this callstack key
             */
            size_t hash() const
            {
                fatalError("Please use subclass");
                return 0;

            }

        };

        /**
         * Represents the complete information about one callstack line.
         *
         * Collecting the complete stack information may be slow, but stack trace must be collected at invocation time.
         * So, only a minimum possible set of information should be collected at invocation time. If this stack inforamtion is indeed used later, more complete inforamtion should be parsed lazily.
         *
         */
        class FrameExtra
        {
        public:
            
            /**
             * Convert the callstack inforamtion into a string.
             */
            std::string toString()
            {
                //fatalError("Abstract interface. Please override this function in subclass");
                return "";
            }

            /**
             * Output the callstack information into std::ostream
             */
            void print(std::ostream &os)
            {
                //fatalError("Abstract interface. Please override this function in subclass");
            }

            
        };


        /**
         * Make callstack function hashable by using its hash() function.
         * T can either be CCallStack
        */
        template<typename T>
        struct Hasher
        {
            size_t operator()(const T& hashable) const
            {
                return hashable.hash();
            }
        };
        
         template<typename T>
        struct HasherPtr
        {
            size_t operator()(const T* hashable) const
            {
                return hashable->hash();
            }
        };
        


        /**
         * Make callstack::native::FrameKey comparable in STL.
         */
        template<typename T>
        struct Comparater
        {
            size_t operator()(const T& key1, const T& key2) const
            {
                return key1==key2;
            }
        };

        template<typename T>
        struct ComparaterPtr
        {
            size_t operator()(const T* key1, const T* key2) const
            {
                return key1->operator==(*key2);
            }
        };

    }

    /**
     * C/C++ callstack
     */
    namespace callstack::native
    {
        class FrameExtra: public callstack::FrameExtra
        {
        public:
            bool valid=false;
            //https://stackoverflow.com/questions/3329956/do-stl-iterators-guarantee-validity-after-collection-was-changed
            std::string fileName; //Save the space of saving multiple executable strings
            std::string symbolName; //Save the space of saving multiple executable strings

            ptrdiff_t offset=0;
            char sign='\0';
            std::string toStringCache; //Since this object is parsed in constructor and will remain unchanged forever, we can safely cache the output in this so that toString will run faster when the same address popup in many different callstacks.


            FrameExtra(void* address);

            /**
             * Convert the callstack inforamtion into a string.
             */
            std::string toString();

            /**
             * Output the callstack information into std::ostream
             */
            void print(std::ostream& os);

        };


        /**
         * This class is similar to callstack::PointerFrameKey, but adds a "version" field.
         * This FrameKey is specifically deisgned to support C++ callstacks because C++ libraries can be unloaded and the address might be reused.
         * The "version" field helps to distinguish re-used address after library unloading.
         * Generally, if two callstack::native::PointerFrameKey (eg: A and B) have the same address but different version, then A!=B. 
         * However there is a special case, if both of the "version" field of A and B is UNSPECIFIED_LATEST_VERSION, then operator== only compares address field. 
         * 
         * This design is based on actual use cases of this class. First, the user of this class needs to lookup callstack::native::FrameExtra by address since there is only address information at the stack unwinding time.
         * "version" is unknown at the stack unwinding time because "ProcInfoParser::loadingCounter" can only be found through a binary search, which is too expensive.
         * Second, MLInsight's ProcInfoParser will pre-parsing "extra" field right after library unloads. At this time, the FrameExtra's "version" field will be set to "ProcInfoParser::loadingCounter" and can be used to distinguish new symbols from the old ones. 
         * "FrameKey" is supposed to be lightweight and expensive fields should be parsed lazily and stored in extraInfo. At first, it seems odd to have a lazily parsed "version" field in callstack::native::PointerFrameKey. But notice that we mandates that callstack should be only compared using FrameKey (a.k.a light-weight inforamtion). Considering the aforementioned usecase, putting "version" field in this class is the best option.
        */
        class PointerFrameKey : public callstack::FrameKey<void*,FrameExtra>
        {
        public:
            ssize_t version;
            static const ssize_t UNSPECIFIED_LATEST_VERSION=-10;
        public:
            
            // The version of elem. If library is unloaded, then the address maybe reused. So one address may corresponds to multiple FrameKey-FrameExtra pairs.
            PointerFrameKey() = default;
            inline PointerFrameKey(void *address, FrameExtra *extra) : callstack::FrameKey<void*, FrameExtra>(address,extra),version(UNSPECIFIED_LATEST_VERSION)
            {
            }

            /**
             * Compare callstack keys
             */
            inline bool operator==(const PointerFrameKey &rho) const
            {
                if(this->version==UNSPECIFIED_LATEST_VERSION && rho.version==UNSPECIFIED_LATEST_VERSION){
                    //Useful when finding FrameKey with the address collected from backtrace.
                    return this->key == rho.key;
                }else{
                    //Useful when comparing saved callstack
                    return this->key == rho.key && this->version == rho.version;
                }
                
            }

            /**
             * Returns the hash of the this callstack key.
             */
            inline size_t hash() const
            {
                return std::hash<void *>()(this->key);
            }

        };
    }

    typedef callstack::native::FrameExtra CFrameExtra_t;
    typedef callstack::native::PointerFrameKey CFrameKey_t;
    extern std::atomic<ssize_t> globalCallStackIdCounter; //This variable 

    class CCallStack : public callstack::CallStack<CFrameKey_t, CPP_CALL_STACK_LEVEL>
    {
    public:
        /*
        * Store the index of CFrameKey_t stored in this->array whose version is UNSPECIFIED_LATEST_VERSION. This helps other parts of the code to only parse not yet lazyily parsed CFrameKey_t and ignore already parsed ones.
        * 
        */
        std::vector<ssize_t> notParsedFrameKeyIndexes; 

        std::string strCache;


    public:
        std::string toString();
        
        void print(std::ostream &output);

        /**
         * Parse complete callstack information of a specific CFrameKey_t.
         * For FrameKey corresponding to an unloaded library, HookInstaller should call this function to fill CFrameKey_t::elem and preserve information right after library unloads.
         * @param notParsedFrameKeyIndexIter An iterator from CCallStack::notParsedFrameKeyIndexes.
         * @param newVersion If library is unloaded, then this field should be set to the loadingCounter in ProcInfoParser. Otherwise, if the library is unloaded this value should still be set to UNSPECIFIED_LATEST_VERSION.
         */
        void parseLine(std::vector<ssize_t>::iterator notParsedFrameKeyIndexIter, ssize_t newVersion=CFrameKey_t::UNSPECIFIED_LATEST_VERSION);

        /**
         * Lazy parse everything inside this->array.
         * This is called at the time where the output is needed to be reported. eg: Program execution end.
         * Code that handles library unloading should call parseLine function to modify a specific line.
        */
        void parseAll();

        void snapshot();

        size_t hash() const;
    };


    extern std::unordered_map<CCallStack*,ssize_t,callstack::HasherPtr<CCallStack>,callstack::ComparaterPtr<CCallStack>>* cCallStackRegistery; //Map CCallStack to an ID
    extern ObjectPoolHeap<CCallStack>* cCallStackHeap;

};
#endif