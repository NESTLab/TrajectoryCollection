/* Include the controller definition */
#include "khepera_mixed.h"
/* Function definitions for XML parsing */
#include <argos3/core/utility/configuration/argos_configuration.h>
#include <argos3/core/utility/logging/argos_log.h>

const Real FLOCKING_1 = -1.0;
const Real FLOCKING_2 = 1.0;
const Real DIFFUSION_Y = -1.0;
const CVector2 FLOCKING_GOAL_1 = CVector2(-0.95, -2.5);
const CVector2 FLOCKING_GOAL_2 = CVector2(1.05, 2.5);

/****************************************/
/****************************************/

CKheperaMixed::CKheperaMixed() :
   m_pcWheels(NULL),
   m_pcProximity(NULL),
   m_fWheelVelocity(2.5f),
   m_pcLEDs(NULL),
   m_pcLight(NULL),
   m_pcRABAct(NULL),
   m_pcRABSens(NULL),
   m_pcGround(NULL),
   m_pcRNG(NULL),
   m_pcPos(NULL),
   m_unRId(0) {}

CKheperaMixed::SFoodData::SFoodData() :
   HasFoodItem(false),
   FoodItemIdx(0),
   TotalFoodItems(0) {}

void CKheperaMixed::SFoodData::Reset() {
   HasFoodItem = false;
   FoodItemIdx = 0;
   TotalFoodItems = 0;
}

CKheperaMixed::SDiffusionParams::SDiffusionParams() :
   GoStraightAngleRange(CRadians(-1.0f), CRadians(1.0f)) {}

/****************************************/
/****************************************/

void CKheperaMixed::SDiffusionParams::Init(TConfigurationNode& t_node) {
   try {
      CRange<CDegrees> cGoStraightAngleRangeDegrees(CDegrees(-10.0f), CDegrees(10.0f));
      GetNodeAttribute(t_node, "go_straight_angle_range", cGoStraightAngleRangeDegrees);
      GoStraightAngleRange.Set(ToRadians(cGoStraightAngleRangeDegrees.GetMin()),
                               ToRadians(cGoStraightAngleRangeDegrees.GetMax()));
      GetNodeAttribute(t_node, "delta", Delta);
   }
   catch(CARGoSException& ex) {
      THROW_ARGOSEXCEPTION_NESTED("Error initializing controller diffusion parameters.", ex);
   }
}

void CKheperaMixed::SWheelTurningParams::Init(TConfigurationNode& t_node) {
   try {
      TurningMechanism = NO_TURN;
      CDegrees cAngle;
      GetNodeAttribute(t_node, "hard_turn_angle_threshold", cAngle);
      HardTurnOnAngleThreshold = ToRadians(cAngle);
      GetNodeAttribute(t_node, "soft_turn_angle_threshold", cAngle);
      SoftTurnOnAngleThreshold = ToRadians(cAngle);
      GetNodeAttribute(t_node, "no_turn_angle_threshold", cAngle);
      NoTurnAngleThreshold = ToRadians(cAngle);
      GetNodeAttribute(t_node, "max_speed", MaxSpeed);
   }
   catch(CARGoSException& ex) {
      THROW_ARGOSEXCEPTION_NESTED("Error initializing controller wheel turning parameters.", ex);
   }
}

/****************************************/
/****************************************/

CKheperaMixed::SStateData::SStateData() :
   ProbRange(0.0f, 1.0f) {}

void CKheperaMixed::SStateData::Init(TConfigurationNode& t_node) {
   try {
      GetNodeAttribute(t_node, "initial_rest_to_explore_prob", InitialRestToExploreProb);
      GetNodeAttribute(t_node, "initial_explore_to_rest_prob", InitialExploreToRestProb);
      GetNodeAttribute(t_node, "food_rule_explore_to_rest_delta_prob", FoodRuleExploreToRestDeltaProb);
      GetNodeAttribute(t_node, "food_rule_rest_to_explore_delta_prob", FoodRuleRestToExploreDeltaProb);
      GetNodeAttribute(t_node, "collision_rule_explore_to_rest_delta_prob", CollisionRuleExploreToRestDeltaProb);
      GetNodeAttribute(t_node, "social_rule_rest_to_explore_delta_prob", SocialRuleRestToExploreDeltaProb);
      GetNodeAttribute(t_node, "social_rule_explore_to_rest_delta_prob", SocialRuleExploreToRestDeltaProb);
      GetNodeAttribute(t_node, "minimum_resting_time", MinimumRestingTime);
      GetNodeAttribute(t_node, "minimum_unsuccessful_explore_time", MinimumUnsuccessfulExploreTime);
      GetNodeAttribute(t_node, "minimum_search_for_place_in_nest_time", MinimumSearchForPlaceInNestTime);

   }
   catch(CARGoSException& ex) {
      THROW_ARGOSEXCEPTION_NESTED("Error initializing controller state parameters.", ex);
   }
}

void CKheperaMixed::SStateData::Reset() {
   State = STATE_RESTING;
   InNest = true;
   RestToExploreProb = InitialRestToExploreProb;
   ExploreToRestProb = InitialExploreToRestProb;
   TimeExploringUnsuccessfully = 0;
   /* Initially the robot is resting, and by setting RestingTime to
      MinimumRestingTime we force the robots to make a decision at the
      experiment start. If instead we set RestingTime to zero, we would
      have to wait till RestingTime reaches MinimumRestingTime before
      something happens, which is just a waste of time. */
   TimeRested = MinimumRestingTime;
   TimeSearchingForPlaceInNest = 0;
}

/****************************************/
/****************************************/

void CKheperaMixed::SFlockingInteractionParams::Init(TConfigurationNode& t_node) {
   try {
      GetNodeAttribute(t_node, "target_distance", TargetDistance);
      GetNodeAttribute(t_node, "gain", Gain);
      GetNodeAttribute(t_node, "exponent", Exponent);
      GetNodeAttribute(t_node, "max_interaction", MaxInteraction);
   }
   catch(CARGoSException& ex) {
      THROW_ARGOSEXCEPTION_NESTED("Error initializing controller flocking parameters.", ex);
   }
}

/****************************************/
/****************************************/

void CKheperaMixed::Init(TConfigurationNode& t_node) {

   m_pcLEDs      = GetActuator<CCI_LEDsActuator                >("leds"                 );
   m_pcRABAct    = GetActuator<CCI_RangeAndBearingActuator  >("range_and_bearing" );
   m_pcRABSens   = GetSensor  <CCI_RangeAndBearingSensor    >("range_and_bearing" );
   m_pcWheels = GetActuator<CCI_DifferentialSteeringActuator          >("differential_steering");
   m_pcLight  = GetSensor  <CCI_KheperaIVLightSensor                    >("kheperaiv_light");
   m_pcProximity = GetSensor  <CCI_KheperaIVProximitySensor             >("kheperaiv_proximity"    );
   m_pcGround    = GetSensor  <CCI_KheperaIVGroundSensor    >("kheperaiv_ground" );
   m_pcPos    = GetSensor  <CCI_PositioningSensor    >("positioning" );

   /*
    * Parse the config file
    */
   GetNodeAttributeOrDefault(t_node, "velocity", m_fWheelVelocity, m_fWheelVelocity);

   try {
      /* Diffusion algorithm */
      m_sDiffusionParams.Init(GetNode(t_node, "diffusion"));
      /* Wheel turning */
      m_sWheelTurningParams.Init(GetNode(t_node, "wheel_turning"));
      /* Controller state */
      m_sStateData.Init(GetNode(t_node, "state"));
      /* Flocking-related */
      m_sFlockingParams.Init(GetNode(t_node, "flocking"));
   }
   catch(CARGoSException& ex) {
      THROW_ARGOSEXCEPTION_NESTED("Error parsing the controller parameters.", ex);
   }
   /*
    * Initialize other stuff
    */
   /* Create a random number generator. We use the 'argos' category so
      that creation, reset, seeding and cleanup are managed by ARGoS. */
   m_pcRNG = CRandom::CreateRNG("argos");
   Reset();
   /* Update Robot Id*/
   const std::string& strRobotId = GetId();
   m_unRId = static_cast<UInt8>(FromString<int>(strRobotId.substr(2)));
}

/****************************************/
/****************************************/

void CKheperaMixed::ControlStep() {

   const CCI_PositioningSensor::SReading& tPosRead = m_pcPos->GetReading();
   CVector3 cPos = tPosRead.Position;
   CQuaternion cQuat = tPosRead.Orientation;
   CRadians cAngle, cX, cY;
   cQuat.ToEulerAngles(cAngle, cY, cX);

   if(cPos.GetX() < FLOCKING_1 or cPos.GetX() > FLOCKING_2){
      CVector2 cPos2(cPos.GetX(), cPos.GetY());
      SetWheelSpeedsFromVector(VectorToGoal(cPos2, cAngle) + FlockingVector());
      LOG << m_unRId << " flocking " << std::endl;
   }
   else if (cPos.GetY() > DIFFUSION_Y)
   {
      LOG << m_unRId << " diffusing" << std::endl;
      /* Get the highest reading in front of the robot, which corresponds to the closest object */
      const CCI_KheperaIVProximitySensor::TReadings& tProxReads = m_pcProximity->GetReadings();
      UInt32 unMaxReadIdx = 0;
      Real fMaxReadVal = tProxReads[unMaxReadIdx].Value;
      if(fMaxReadVal < tProxReads[1].Value) {
         fMaxReadVal = tProxReads[1].Value;
         unMaxReadIdx = 1;
      }
      if(fMaxReadVal < tProxReads[7].Value) {
         fMaxReadVal = tProxReads[7].Value;
         unMaxReadIdx = 7;
      }
      if(fMaxReadVal < tProxReads[6].Value) {
         fMaxReadVal = tProxReads[6].Value;
         unMaxReadIdx = 6;
      }
      /* Do we have an obstacle in front? */
      if(fMaxReadVal > 0.0f) {
        /* Yes, we do: avoid it */
        if(unMaxReadIdx == 0 || unMaxReadIdx == 1) {
          /* The obstacle is on the left, turn right */
          m_pcWheels->SetLinearVelocity(m_fWheelVelocity, 0.0f);
        }
        else {
          /* The obstacle is on the left, turn right */
          m_pcWheels->SetLinearVelocity(0.0f, m_fWheelVelocity);
        }
      }
      else {
        /* No, we don't: go straight */
         m_pcWheels->SetLinearVelocity(m_fWheelVelocity, m_fWheelVelocity);
      }
   }
   else
   {
      LOG << m_unRId << " foraging" << std::endl;
      switch(m_sStateData.State) {
      case SStateData::STATE_RESTING: {
         Rest();
         break;
      }
      case SStateData::STATE_EXPLORING: {
         Explore();
         break;
      }
      case SStateData::STATE_RETURN_TO_NEST: {
         ReturnToNest();
         break;
      }
      default: {
         LOGERR << "We can't be here, there's a bug!" << std::endl;
      }
   }
   }
   m_pcRABAct->SetData(0, m_unRId);
}

/****************************************/
/****************************************/

void CKheperaMixed::Reset() {
   /* Reset robot state */
   m_sStateData.Reset();
   /* Reset food data */
   m_sFoodData.Reset();
   /* Set LED color */
   m_pcLEDs->SetAllColors(CColor::RED);
   /* Clear up the last exploration result */
   m_eLastExplorationResult = LAST_EXPLORATION_NONE;
   m_pcRABAct->ClearData();
   m_pcRABAct->SetData(0, m_unRId);
   m_pcRABAct->SetData(1, LAST_EXPLORATION_NONE);
}

/****************************************/
/****************************************/

void CKheperaMixed::UpdateState() {
   /* Reset state flags */
   m_sStateData.InNest = false;
   /* Read stuff from the ground sensor */
   const CCI_KheperaIVGroundSensor::TReadings& tGroundReads = m_pcGround->GetReadings();
   /*
    * You can say whether you are in the nest by checking the ground sensor
    * placed close to the wheel motors. It returns a value between 0 and 1.
    * It is 1 when the robot is on a white area, it is 0 when the robot
    * is on a black area and it is around 0.5 when the robot is on a gray
    * area. 
    * The foot-bot has 4 sensors like this, two in the front
    * (corresponding to readings 0 and 1) and two in the back
    * (corresponding to reading 2 and 3).  Here we want the back sensors
    * (readings 2 and 3) to tell us whether we are on gray: if so, the
    * robot is completely in the nest, otherwise it's outside.
    */
   if(tGroundReads[2].Value > 0.25f &&
      tGroundReads[2].Value < 0.75f &&
      tGroundReads[3].Value > 0.25f &&
      tGroundReads[3].Value < 0.75f) {
      m_sStateData.InNest = true;
   }
}

/****************************************/
/****************************************/

CVector2 CKheperaMixed::CalculateVectorToLight() {
   /* Get readings from light sensor */
   const CCI_KheperaIVLightSensor::TReadings& tLightReads = m_pcLight->GetReadings();
   /* Sum them together */
   CVector2 cAccumulator;
   for(size_t i = 0; i < tLightReads.size(); ++i) {
      cAccumulator += CVector2(tLightReads[i].Value, tLightReads[i].Angle);
   }
   /* If the light was perceived, return the vector */
   if(cAccumulator.Length() > 0.0f) {
      return CVector2(1.0f, cAccumulator.Angle());
   }
   /* Otherwise, return zero */
   else {
      return CVector2();
   }
}

/****************************************/
/****************************************/

CVector2 CKheperaMixed::DiffusionVector(bool& b_collision) {
   /* Get readings from proximity sensor */
   const CCI_KheperaIVProximitySensor::TReadings& tProxReads = m_pcProximity->GetReadings();
   /* Sum them together */
   CVector2 cDiffusionVector;
   for(size_t i = 0; i < tProxReads.size(); ++i) {
      cDiffusionVector += CVector2(tProxReads[i].Value, tProxReads[i].Angle);
   }
   /* If the angle of the vector is small enough and the closest obstacle
      is far enough, ignore the vector and go straight, otherwise return
      it */
   if(m_sDiffusionParams.GoStraightAngleRange.WithinMinBoundIncludedMaxBoundIncluded(cDiffusionVector.Angle()) &&
      cDiffusionVector.Length() < m_sDiffusionParams.Delta ) {
      b_collision = false;
      return CVector2::X;
   }
   else {
      b_collision = true;
      cDiffusionVector.Normalize();
      return -cDiffusionVector;
   }
}

/****************************************/
/****************************************/

void CKheperaMixed::SetWheelSpeedsFromVector(const CVector2& c_heading) {
   //if (m_unRId == 1) return;

   /* Get the heading angle */
   CRadians cHeadingAngle = c_heading.Angle().SignedNormalize();
   /* Get the length of the heading vector */
   Real fHeadingLength = c_heading.Length();
   /* Clamp the speed so that it's not greater than MaxSpeed */
   Real fBaseAngularWheelSpeed = Min<Real>(fHeadingLength, m_sWheelTurningParams.MaxSpeed);
   /* State transition logic */
   if(m_sWheelTurningParams.TurningMechanism == SWheelTurningParams::HARD_TURN) {
      if(Abs(cHeadingAngle) <= m_sWheelTurningParams.SoftTurnOnAngleThreshold) {
         m_sWheelTurningParams.TurningMechanism = SWheelTurningParams::SOFT_TURN;
      }
   }
   if(m_sWheelTurningParams.TurningMechanism == SWheelTurningParams::SOFT_TURN) {
      if(Abs(cHeadingAngle) > m_sWheelTurningParams.HardTurnOnAngleThreshold) {
         m_sWheelTurningParams.TurningMechanism = SWheelTurningParams::HARD_TURN;
      }
      else if(Abs(cHeadingAngle) <= m_sWheelTurningParams.NoTurnAngleThreshold) {
         m_sWheelTurningParams.TurningMechanism = SWheelTurningParams::NO_TURN;
      }
   }
   if(m_sWheelTurningParams.TurningMechanism == SWheelTurningParams::NO_TURN) {
      if(Abs(cHeadingAngle) > m_sWheelTurningParams.HardTurnOnAngleThreshold) {
         m_sWheelTurningParams.TurningMechanism = SWheelTurningParams::HARD_TURN;
      }
      else if(Abs(cHeadingAngle) > m_sWheelTurningParams.NoTurnAngleThreshold) {
         m_sWheelTurningParams.TurningMechanism = SWheelTurningParams::SOFT_TURN;
      }
   }
   /* Wheel speeds based on current turning state */
   Real fSpeed1, fSpeed2;
   switch(m_sWheelTurningParams.TurningMechanism) {
      case SWheelTurningParams::NO_TURN: {
         /* Just go straight */
         fSpeed1 = fBaseAngularWheelSpeed;
         fSpeed2 = fBaseAngularWheelSpeed;
         break;
      }
      case SWheelTurningParams::SOFT_TURN: {
         /* Both wheels go straight, but one is faster than the other */
         Real fSpeedFactor = (m_sWheelTurningParams.HardTurnOnAngleThreshold - Abs(cHeadingAngle)) / m_sWheelTurningParams.HardTurnOnAngleThreshold;
         fSpeed1 = fBaseAngularWheelSpeed - fBaseAngularWheelSpeed * (1.0 - fSpeedFactor);
         fSpeed2 = fBaseAngularWheelSpeed + fBaseAngularWheelSpeed * (1.0 - fSpeedFactor);
         break;
      }
      case SWheelTurningParams::HARD_TURN: {
         /* Opposite wheel speeds */
         fSpeed1 = -m_sWheelTurningParams.MaxSpeed;
         fSpeed2 =  m_sWheelTurningParams.MaxSpeed;
         break;
      }
   }
   /* Apply the calculated speeds to the appropriate wheels */
   Real fLeftWheelSpeed, fRightWheelSpeed;
   if(cHeadingAngle > CRadians::ZERO) {
      /* Turn Left */
      fLeftWheelSpeed  = fSpeed1;
      fRightWheelSpeed = fSpeed2;
   }
   else {
      /* Turn Right */
      fLeftWheelSpeed  = fSpeed2;
      fRightWheelSpeed = fSpeed1;
   }
   /* Finally, set the wheel speeds */
   m_pcWheels->SetLinearVelocity(fLeftWheelSpeed, fRightWheelSpeed);
}

/****************************************/
/****************************************/

void CKheperaMixed::Rest() {
   /* If we have stayed here enough, probabilistically switch to
    * 'exploring' */
   if(m_sStateData.TimeRested > m_sStateData.MinimumRestingTime &&
      m_pcRNG->Uniform(m_sStateData.ProbRange) < m_sStateData.RestToExploreProb) {
      m_pcLEDs->SetAllColors(CColor::GREEN);
      m_sStateData.State = SStateData::STATE_EXPLORING;
      m_sStateData.TimeRested = 0;
   }
   else {
      ++m_sStateData.TimeRested;
      /* Be sure not to send the last exploration result multiple times */
      if(m_sStateData.TimeRested == 1) {
         m_pcRABAct->SetData(1, LAST_EXPLORATION_NONE);
      }
      /*
       * Social rule: listen to what other people have found and modify
       * probabilities accordingly
       */
      const CCI_RangeAndBearingSensor::TReadings& tPackets = m_pcRABSens->GetReadings();
      for(size_t i = 0; i < tPackets.size(); ++i) {
         switch(tPackets[i].Data[1]) {
            case LAST_EXPLORATION_SUCCESSFUL: {
               m_sStateData.RestToExploreProb += m_sStateData.SocialRuleRestToExploreDeltaProb;
               m_sStateData.ProbRange.TruncValue(m_sStateData.RestToExploreProb);
               m_sStateData.ExploreToRestProb -= m_sStateData.SocialRuleExploreToRestDeltaProb;
               m_sStateData.ProbRange.TruncValue(m_sStateData.ExploreToRestProb);
               break;
            }
            case LAST_EXPLORATION_UNSUCCESSFUL: {
               m_sStateData.ExploreToRestProb += m_sStateData.SocialRuleExploreToRestDeltaProb;
               m_sStateData.ProbRange.TruncValue(m_sStateData.ExploreToRestProb);
               m_sStateData.RestToExploreProb -= m_sStateData.SocialRuleRestToExploreDeltaProb;
               m_sStateData.ProbRange.TruncValue(m_sStateData.RestToExploreProb);
               break;
            }
         }
      }
   }
}

/****************************************/
/****************************************/

void CKheperaMixed::Explore() {
   /* We switch to 'return to nest' in two situations:
    * 1. if we have a food item
    * 2. if we have not found a food item for some time;
    *    in this case, the switch is probabilistic
    */
   bool bReturnToNest(false);
   /*
    * Test the first condition: have we found a food item?
    * NOTE: the food data is updated by the loop functions, so
    * here we just need to read it
    */
   if(m_sFoodData.HasFoodItem) {
      /* Apply the food rule, decreasing ExploreToRestProb and increasing
       * RestToExploreProb */
      m_sStateData.ExploreToRestProb -= m_sStateData.FoodRuleExploreToRestDeltaProb;
      m_sStateData.ProbRange.TruncValue(m_sStateData.ExploreToRestProb);
      m_sStateData.RestToExploreProb += m_sStateData.FoodRuleRestToExploreDeltaProb;
      m_sStateData.ProbRange.TruncValue(m_sStateData.RestToExploreProb);
      /* Store the result of the expedition */
      m_eLastExplorationResult = LAST_EXPLORATION_SUCCESSFUL;
      /* Switch to 'return to nest' */
      bReturnToNest = true;
   }
   /* Test the second condition: we probabilistically switch to 'return to
    * nest' if we have been wandering for some time and found nothing */
   else if(m_sStateData.TimeExploringUnsuccessfully > m_sStateData.MinimumUnsuccessfulExploreTime) {
      if (m_pcRNG->Uniform(m_sStateData.ProbRange) < m_sStateData.ExploreToRestProb) {
         /* Store the result of the expedition */
         m_eLastExplorationResult = LAST_EXPLORATION_UNSUCCESSFUL;
         /* Switch to 'return to nest' */
         bReturnToNest = true;
      }
      else {
         /* Apply the food rule, increasing ExploreToRestProb and
          * decreasing RestToExploreProb */
         m_sStateData.ExploreToRestProb += m_sStateData.FoodRuleExploreToRestDeltaProb;
         m_sStateData.ProbRange.TruncValue(m_sStateData.ExploreToRestProb);
         m_sStateData.RestToExploreProb -= m_sStateData.FoodRuleRestToExploreDeltaProb;
         m_sStateData.ProbRange.TruncValue(m_sStateData.RestToExploreProb);
      }
   }
   /* So, do we return to the nest now? */
   if(bReturnToNest) {
      /* Yes, we do! */
      m_sStateData.TimeExploringUnsuccessfully = 0;
      m_sStateData.TimeSearchingForPlaceInNest = 0;
      m_pcLEDs->SetAllColors(CColor::BLUE);
      m_sStateData.State = SStateData::STATE_RETURN_TO_NEST;
   }
   else {
      /* No, perform the actual exploration */
      ++m_sStateData.TimeExploringUnsuccessfully;
      UpdateState();
      /* Get the diffusion vector to perform obstacle avoidance */
      bool bCollision;
      CVector2 cDiffusion = DiffusionVector(bCollision);
      /* Apply the collision rule, if a collision avoidance happened */
      if(bCollision) {
         /* Collision avoidance happened, increase ExploreToRestProb and
          * decrease RestToExploreProb */
         m_sStateData.ExploreToRestProb += m_sStateData.CollisionRuleExploreToRestDeltaProb;
         m_sStateData.ProbRange.TruncValue(m_sStateData.ExploreToRestProb);
         m_sStateData.RestToExploreProb -= m_sStateData.CollisionRuleExploreToRestDeltaProb;
         m_sStateData.ProbRange.TruncValue(m_sStateData.RestToExploreProb);
      }
      /*
       * If we are in the nest, we combine antiphototaxis with obstacle
       * avoidance
       * Outside the nest, we just use the diffusion vector
       */
      if(m_sStateData.InNest) {
         /*
          * The vector returned by CalculateVectorToLight() points to
          * the light. Thus, the minus sign is because we want to go away
          * from the light.
          */
         SetWheelSpeedsFromVector(
            m_sWheelTurningParams.MaxSpeed * cDiffusion -
            m_sWheelTurningParams.MaxSpeed * 0.25f * CalculateVectorToLight());
      }
      else {
         /* Use the diffusion vector only */
         SetWheelSpeedsFromVector(m_sWheelTurningParams.MaxSpeed * cDiffusion);
      }
   }
}

/****************************************/
/****************************************/

void CKheperaMixed::ReturnToNest() {
   /* As soon as you get to the nest, switch to 'resting' */
   UpdateState();
   /* Are we in the nest? */
   if(m_sStateData.InNest) {
      /* Have we looked for a place long enough? */
      if(m_sStateData.TimeSearchingForPlaceInNest > m_sStateData.MinimumSearchForPlaceInNestTime) {
         /* Yes, stop the wheels... */
         m_pcWheels->SetLinearVelocity(0.0f, 0.0f);
         /* Tell people about the last exploration attempt */
         m_pcRABAct->SetData(1, m_eLastExplorationResult);
         /* ... and switch to state 'resting' */
         m_pcLEDs->SetAllColors(CColor::RED);
         m_sStateData.State = SStateData::STATE_RESTING;
         m_sStateData.TimeSearchingForPlaceInNest = 0;
         m_eLastExplorationResult = LAST_EXPLORATION_NONE;
         return;
      }
      else {
         /* No, keep looking */
         ++m_sStateData.TimeSearchingForPlaceInNest;
      }
   }
   else {
      /* Still outside the nest */
      m_sStateData.TimeSearchingForPlaceInNest = 0;
   }
   /* Keep going */
   bool bCollision;
   SetWheelSpeedsFromVector(
      m_sWheelTurningParams.MaxSpeed * DiffusionVector(bCollision) +
      m_sWheelTurningParams.MaxSpeed * CalculateVectorToLight());
}

/****************************************/
/****************************************/

Real CKheperaMixed::SFlockingInteractionParams::GeneralizedLennardJones(Real f_distance) {
   Real fNormDistExp = ::pow(TargetDistance / f_distance, Exponent);
   return -Gain / f_distance * (fNormDistExp * fNormDistExp - fNormDistExp);
}

/****************************************/
/****************************************/

CVector2 CKheperaMixed::VectorToGoal(const CVector2& c_pos, const CRadians c_angle) {
   /* Get light readings */
   /* Calculate a normalized vector that points to the closest light */
   CVector2 cAccum;
   if (c_pos.GetX() < FLOCKING_1)
   {
      CVector2 cDiff = FLOCKING_GOAL_1 - c_pos;
      Real fDistance = cDiff.Length();
      CRadians fVecAngle = ATan2(FLOCKING_GOAL_1.GetY() - c_pos.GetY(), FLOCKING_GOAL_1.GetX() - c_pos.GetX()) - c_angle;
      cAccum.FromPolarCoordinates(fDistance , fVecAngle); 
   }
   else if (c_pos.GetX() > FLOCKING_2)
   {
      CVector2 cDiff = FLOCKING_GOAL_2 - c_pos;
      Real fDistance = cDiff.Length();
      CRadians fVecAngle = ATan2(FLOCKING_GOAL_2.GetY() - c_pos.GetY(), FLOCKING_GOAL_2.GetX() - c_pos.GetX()) - c_angle;
      cAccum.FromPolarCoordinates(fDistance , fVecAngle); 
   }
   if(cAccum.Length() > 0.0f) {
       // Make the vector long as 1/4 of the max speed 
      cAccum.Normalize();
      cAccum *= 0.25f * m_sWheelTurningParams.MaxSpeed;
   }
   return cAccum;
}

/****************************************/
/****************************************/

CVector2 CKheperaMixed::FlockingVector() {
   /* Get RAB messages from nearby kheperas */
   const CCI_RangeAndBearingSensor::TReadings& tMsgs = m_pcRABSens->GetReadings();
   /* Go through them to calculate the flocking interaction vector */
   if(! tMsgs.empty()) {
      /* This will contain the final interaction vector */
      CVector2 cAccum;
      /* Used to calculate the vector length of each neighbor's contribution */
      Real fLJ;
      /* A counter for the neighbors in state flock */
      UInt32 unPeers = 0;
      for(size_t i = 0; i < tMsgs.size(); ++i) {
         if(tMsgs[i].Data[0] > 0) {
            /*
             * Take the message sender range and horizontal bearing
             * With the range, calculate the Lennard-Jones interaction force
             * Form a 2D vector with the interaction force and the bearing
             * Sum such vector to the accumulator
             */
            /* Calculate LJ */
            fLJ = m_sFlockingParams.GeneralizedLennardJones(tMsgs[i].Range);
            /* Sum to accumulator */
            cAccum += CVector2(fLJ,
                               tMsgs[i].HorizontalBearing);
            /* Count one more flocking neighbor */
            ++unPeers;
         }
      }
      if(unPeers > 0) {
         /* Divide the accumulator by the number of flocking neighbors */
         cAccum /= unPeers;
         /* Limit the interaction force */
         if(cAccum.Length() > m_sFlockingParams.MaxInteraction) {
            cAccum.Normalize();
            cAccum *= m_sFlockingParams.MaxInteraction;
         }
      }
      /* All done */
      return cAccum;
   }
   else {
      /* No messages received, no interaction */
      return CVector2();
   }
}

/****************************************/
/****************************************/

/*
 * This statement notifies ARGoS of the existence of the controller.
 * It binds the class passed as first argument to the string passed as second argument.
 * The string is then usable in the XML configuration file to refer to this controller.
 * When ARGoS reads that string in the XML file, it knows which controller class to instantiate.
 * See also the XML configuration files for an example of how this is used.
 */
REGISTER_CONTROLLER(CKheperaMixed, "khepera_mixed_controller")
