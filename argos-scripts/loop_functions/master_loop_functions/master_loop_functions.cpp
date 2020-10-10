#include "master_loop_functions.h"

#include <argos3/core/utility/plugins/dynamic_loading.h>

void CMasterLoopFunctions::Init(TConfigurationNode& t_tree){
    m_cForagingLF = nullptr;
    m_cMixedLF = nullptr;
    TConfigurationNodeIterator cCurrNode;
    // iterate through all nodes
    for(cCurrNode = cCurrNode.begin(&t_tree);
          cCurrNode != cCurrNode.end();
          ++cCurrNode){
        if(cCurrNode->Value() == "sub_loop_function" || cCurrNode->Value() == "sub_loop_functions"){
            try {
                std::string strLibrary, strLabel;
                UInt8 unPriority;

                // get generic information
                GetNodeAttributeOrDefault(*cCurrNode, "library", strLibrary, strLibrary);
                GetNodeAttribute(*cCurrNode, "label", strLabel);
                GetNodeAttributeOrDefault(*cCurrNode, "priority", unPriority, 0);

                if(! strLibrary.empty()) {
                    CDynamicLoading::LoadLibrary(strLibrary);
                }

                // create the new fuction and initialize it
                CLoopFunctions *cNewFunction = CFactory<CLoopFunctions>::New(strLabel);
                cNewFunction->Init(*cCurrNode);

                //push teh new loop function on the to vector
                m_cLoopFunctions.emplace_back(SLoopFunctionTuple(cNewFunction, unPriority, strLabel));

                if(strLabel == "foraging_loop_functions")
                   m_cForagingLF = cNewFunction;
                else if(strLabel == "mixed_loop_functions")
                   m_cMixedLF = cNewFunction;
            }
                catch(CARGoSException& ex) {
                    THROW_ARGOSEXCEPTION_NESTED("Error initializing master functions", ex);
            }
        }
    }

    std::sort(m_cLoopFunctions.begin(), m_cLoopFunctions.end());
}

/****************************************/
/****************************************/

void CMasterLoopFunctions::Reset(){
    for(TLoopFunctionTuple tCurrentLF: m_cLoopFunctions){
        tCurrentLF.pcLoopFunction->Reset();
    }
}

/****************************************/
/****************************************/

void CMasterLoopFunctions::Destroy(){
    for(TLoopFunctionTuple tCurrentLF: m_cLoopFunctions){
        tCurrentLF.pcLoopFunction->Destroy();
        delete tCurrentLF.pcLoopFunction;
    }
}

/****************************************/
/****************************************/

void CMasterLoopFunctions::PreStep(){
    for(TLoopFunctionTuple tCurrentLF: m_cLoopFunctions){
        tCurrentLF.pcLoopFunction->PreStep();
    }
}

/****************************************/
/****************************************/

void CMasterLoopFunctions::PostStep(){
    for(TLoopFunctionTuple tCurrentLF: m_cLoopFunctions){
        tCurrentLF.pcLoopFunction->PostStep();
    }
}

/****************************************/
/****************************************/

bool CMasterLoopFunctions::IsExperimentFinished(){
    bool bFinished = false;
    for(TLoopFunctionTuple tCurrentLF: m_cLoopFunctions){
        bFinished &= tCurrentLF.pcLoopFunction->IsExperimentFinished();
    }
    return bFinished;
}

/****************************************/
/****************************************/

void CMasterLoopFunctions::PostExperiment(){
    for(TLoopFunctionTuple tCurrentLF: m_cLoopFunctions){
        tCurrentLF.pcLoopFunction->PostExperiment();
    }
}

/****************************************/
/****************************************/

CColor CMasterLoopFunctions::GetFloorColor(const CVector2& c_position_on_plane) {
   if(m_cForagingLF != nullptr) return m_cForagingLF->GetFloorColor(c_position_on_plane);
   if(m_cMixedLF != nullptr)    return m_cMixedLF->GetFloorColor(c_position_on_plane);
   return CColor::WHITE;
}

/****************************************/
/****************************************/

CLoopFunctions& CMasterLoopFunctions::GetLoopFunction(std::string label){
    LOG << "There are " << m_cLoopFunctions.size() << " subl loop functions" << std::endl;
    for(TLoopFunctionTuple tCurrentLF: m_cLoopFunctions){
        if(tCurrentLF.strLabel == label)
            return *(tCurrentLF.pcLoopFunction);
        LOG << tCurrentLF.strLabel << " does not match " << label << std::endl;
    }
    LOG.Flush();
    THROW_ARGOSEXCEPTION("Loop Function with label '" << label 
                            <<"' not found." << std::endl;);
}

REGISTER_LOOP_FUNCTIONS(CMasterLoopFunctions, "MasterLoopFunctions");
