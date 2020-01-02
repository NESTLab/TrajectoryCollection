#ifndef MASTER_LOOP_FUNCTION_H
#define MASTER_LOOP_FUNCTION_H

#include <vector>

#include <argos3/core/simulator/loop_functions.h>

using namespace argos;


// Data structure for holding a single rule unPriority denotes how important 
// a rule is and what presidence it takes. Priority defaults to 0, with higher 
// rules having higher priority and being executed first. 
// The largest value for priority is 254.
// Rules with the same priority do not have their order garenteed and 
// code that relies on such an ordering is considered buggy.  
typedef struct SLoopFunctionTuple{
    CLoopFunctions *pcLoopFunction;
    UInt8 unPriority;
    std::string strLabel;

    SLoopFunctionTuple(CLoopFunctions *pc_function, 
                       UInt8 un_priority, 
                       std::string label):
        pcLoopFunction(pc_function),
        unPriority(un_priority),
        strLabel(std::move(label)){}
    bool operator < (const SLoopFunctionTuple& sOther) const{
        return (unPriority < sOther.unPriority);
    }

} TLoopFunctionTuple;

class CMasterLoopFunctions : public CLoopFunctions{

public:

    void Init(TConfigurationNode& t_tree) override;

    void Reset() override;

    void Destroy() override;

    void PreStep() override;

    void PostStep() override;

    bool IsExperimentFinished() override;

    void PostExperiment() override;
    
    CLoopFunctions& GetLoopFunction(std::string label);

    template <typename T>
    inline static T& GetLoopFunction (const std::string& loop_function)  {
        auto* pcMLF = dynamic_cast<CMasterLoopFunctions*>( &CSimulator::GetInstance().GetLoopFunctions());
        
        if(pcMLF != nullptr){
            return dynamic_cast<T&>(pcMLF->GetLoopFunction(loop_function));
        }
        return dynamic_cast<T&>(CSimulator::GetInstance().GetLoopFunctions());\
    } 

private:
    std::vector<TLoopFunctionTuple> m_cLoopFunctions;

};


#endif
