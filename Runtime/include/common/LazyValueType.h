
#ifndef MLINSIGHT_LAZYVALUETYPE_H
#define MLINSIGHT_LAZYVALUETYPE_H
#include <cstdlib>
#include "common/Tool.h"

namespace mlinsight{
    /**
     * Value type that supports lazy construct.
     */
    template<typename VALUE_TYPE>
    class LazyConstructValue {
    protected:
        char value[sizeof(VALUE_TYPE)]; //If initialized, value will be of type VALUE_TYPE
        bool valueConstructed=false;
    public:
        LazyConstructValue()=default;

        /**
        * Copy and swap idiom: Copy constructor
        */
        LazyConstructValue(const LazyConstructValue& rho):valueConstructed(rho.valueConstructed){
            if(rho.valueConstructed){
                auto* thisValuePointer=reinterpret_cast<VALUE_TYPE*>(value);
                auto& rhoValue=*reinterpret_cast<const VALUE_TYPE*>(rho.value);
                //Copy value object using the VALUE_TYPE's copy constructor
                new (thisValuePointer) VALUE_TYPE(rhoValue);
            }else{
                //If value is not constructed, then we do not need to construct either.
            }
        }

        /**
        * Copy and swap idiom: Move constructor.
        */
        LazyConstructValue(LazyConstructValue&& rho):valueConstructed(rho.valueConstructed){
            if(rho.valueConstructed){
                auto* thisValuePointer=reinterpret_cast<VALUE_TYPE*>(value);
                auto& rhoValue=*reinterpret_cast<VALUE_TYPE*>(rho.value);
                //Copy value object using the VALUE_TYPE's move constructor
                new (thisValuePointer) VALUE_TYPE(std::move(rhoValue));
            }else{
                //If value is not constructed, then we do not need move constructor either.
            }
        }

        template<typename... Args>
        inline void constructValue(Args&&... args) {
            assert(!valueConstructed);
            auto* thisValuePointer=reinterpret_cast<VALUE_TYPE*>(value);
            new (thisValuePointer) VALUE_TYPE(std::forward<Args>(args)...);
            valueConstructed=true;
        }

        inline void destructValue(){
            if(valueConstructed){
                auto* valuePointer=reinterpret_cast<VALUE_TYPE*>(value);
                valuePointer->~VALUE_TYPE();
                valueConstructed=false;
            }
        }
        bool isValueConstructed(){
            return valueConstructed;
        }
        VALUE_TYPE& getConstructedValue(){
            assert(valueConstructed);
            return *(reinterpret_cast<VALUE_TYPE*>(value));
        }

        /**
        * Copy and swap idiom: Copy assignment.
        */
        LazyConstructValue& operator=(const LazyConstructValue& rho){
            if(this!=&rho){
                LazyConstructValue tempObject(rho);
                swap(*this,tempObject);
            }
            return *this;
        }

        /**
        * Copy and swap idiom: Move assignment.
        */
        LazyConstructValue& operator=(LazyConstructValue&& rho){
            swap(*this,rho);
            return *this;
        }

        /**
        * Copy and swap idiom: swap.
        */
        friend void swap(LazyConstructValue& lho,LazyConstructValue& rho){
            using std::swap;
            auto& lhoValue=*reinterpret_cast<VALUE_TYPE*>(lho.value);
            auto& rhoValue=*reinterpret_cast<VALUE_TYPE*>(rho.value);

            swap(lho.valueConstructed,rho.valueConstructed);
            swap(lhoValue,rhoValue);
        }

        bool operator==(const LazyConstructValue& rho){
            if(!valueConstructed && !rho.valueConstructed){
                //Both listEntries are empty
                return true;
            }

            if(valueConstructed && rho.valueConstructed){
                auto& thisValue=reinterpret_cast<VALUE_TYPE&>(value);
                const auto& thatValue=(*reinterpret_cast<const VALUE_TYPE*>(rho.value));
                //Delegate compare operation to VALUE_TYPE::operator==()
                return thisValue == thatValue;
            }

            //One is initialized the other is not, so not equal.
            //We cannot compare objects by invoking operator= because objects may not be constructed.
            return false;
        }

        bool operator!=(const LazyConstructValue& rho){
            return !operator==(rho);
        }

        ~LazyConstructValue(){
            destructValue();
        }


    };

}
#endif //MLINSIGHT_LAZYVALUETYPE_H
