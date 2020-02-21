#include "foraging_qt_user_functions.h"
#include <controllers/khepera_mixed/khepera_mixed.h>
#include <argos3/core/simulator/entity/controllable_entity.h>

using namespace argos;

/****************************************/
/****************************************/

CForagingQTUserFunctions::CForagingQTUserFunctions() {
   RegisterUserFunction<CForagingQTUserFunctions,CKheperaIVEntity>(&CForagingQTUserFunctions::Draw);
}

/****************************************/
/****************************************/

void CForagingQTUserFunctions::Draw(CKheperaIVEntity& c_entity) {
   CKheperaMixed& cController = dynamic_cast<CKheperaMixed&>(c_entity.GetControllableEntity().GetController());
   CKheperaMixed::SFoodData& sFoodData = cController.GetFoodData();
   if(sFoodData.HasFoodItem) {
      DrawCylinder(
         CVector3(0.0f, 0.0f, 0.3f), 
         CQuaternion(),
         0.1f,
         0.05f,
         CColor::BLACK);
   }
}

/****************************************/
/****************************************/

REGISTER_QTOPENGL_USER_FUNCTIONS(CForagingQTUserFunctions, "foraging_qt_user_functions")
