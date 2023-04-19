// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
// V0 Finder task
// ==============
//
// This code loops over positive and negative tracks and finds
// valid V0 candidates from scratch using a certain set of
// minimum (configurable) selection criteria.
//
// It is different than the producer: the producer merely
// loops over an *existing* list of V0s (pos+neg track
// indices) and calculates the corresponding full V0 information
//
// In both cases, any analysis should loop over the "V0Data"
// table as that table contains all information.
//
//

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "DCAFitter/DCAFitterN.h"
#include "ReconstructionDataFormats/Track.h"
#include "Common/Core/RecoDecay.h"
#include "Common/Core/trackUtilities.h"
#include "PWGLF/DataModel/LFStrangenessTables.h"
#include "Common/Core/TrackSelection.h"
#include "Common/DataModel/TrackSelectionTables.h"
#include "Common/DataModel/EventSelection.h"
#include "Common/DataModel/Centrality.h"
#include "Common/DataModel/PIDResponse.h"
//------------------copy from lamdakzerobuilder---------------------
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GeometryManager.h"
#include "DataFormatsParameters/GRPObject.h"
#include <CCDB/BasicCCDBManager.h>
//------------------copy from lamdakzerobuilder---------------------
#include "DataFormatsTPC/BetheBlochAleph.h"

#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TLorentzVector.h>
#include <Math/Vector4D.h>
#include <TPDGCode.h>
#include <TDatabasePDG.h>
#include <cmath>
#include <array>
#include <cstdlib>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;
using namespace ROOT::Math;

//use parameters + cov mat non-propagated, aux info + (extension propagated)
using FullTracksExt = soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksCov, aod::TracksDCA, aod::pidTPCFullPi, aod::pidTPCFullHe>;
using FullTracksExtMC = soa::Join<FullTracksExt, aod::McTrackLabels, aod::pidTPCFullPi, aod::pidTPCFullHe>;
using FullTracksExtIU = soa::Join<aod::TracksIU, aod::TracksExtra, aod::TracksCovIU, aod::TracksDCA, aod::pidTPCFullPi, aod::pidTPCFullHe>;
using FullTracksExtMCIU = soa::Join<FullTracksExtIU, aod::McTrackLabels>;

using MyTracks = FullTracksExt;
using MyTracksIU = FullTracksExtIU;

inline float GetTPCNSigmaHe3(float p, float TPCSignal)
{
  float bg = p/2.80839;
  return  (TPCSignal - o2::tpc::BetheBlochAleph(bg, -9.973f, -18.5543f, 29.5704f, 2.02064f, -3.85076f)) / (TPCSignal*0.0812);
}

namespace o2::aod
{
  namespace v0goodpostracks
  {
    DECLARE_SOA_INDEX_COLUMN_FULL(GoodTrack, goodTrack, int, Tracks, "_GoodTrack");
    DECLARE_SOA_INDEX_COLUMN(Collision, collision);
    DECLARE_SOA_COLUMN(DCAXY, dcaXY, float);
  } // namespace v0goodpostracks
  DECLARE_SOA_TABLE(V0GoodPosTracks, "AOD", "V0GOODPOSTRACKS", o2::soa::Index<>, v0goodpostracks::GoodTrackId, v0goodpostracks::CollisionId, v0goodpostracks::DCAXY);
  namespace v0goodnegtracks
  {
    DECLARE_SOA_INDEX_COLUMN_FULL(GoodTrack, goodTrack, int, Tracks, "_GoodTrack");
    DECLARE_SOA_INDEX_COLUMN(Collision, collision);
    DECLARE_SOA_COLUMN(DCAXY, dcaXY, float);
  } // namespace v0goodnegtracks
  DECLARE_SOA_TABLE(V0GoodNegTracks, "AOD", "V0GOODNEGTRACKS", o2::soa::Index<>, v0goodnegtracks::GoodTrackId, v0goodnegtracks::CollisionId, v0goodnegtracks::DCAXY);
} // namespace o2::aod

struct hypertritonprefilter {
  HistogramRegistry registry{
    "registry",
      {
        {"hCrossedRows", "hCrossedRows", {HistType::kTH1F, {{50, 0.0f, 200.0f}}}},
        {"hGoodTrackCount", "hGoodTrackCount",{HistType::kTH1F, {{4, 0.0f, 4.0f}}}},
        {"hGoodPosTrackCount", "hGoodPosTrackCount",{HistType::kTH1F, {{1, 0.0f, 1.0f}}}},
        {"hGoodNegTrackCount", "hGoodNegTrackCount",{HistType::kTH1F, {{1, 0.0f, 1.0f}}}},
      },
  };

  //change the dca cut for helium3
  Configurable<float> dcanegtopv{"dcanegtopv", .1, "DCA Neg To PV"};
  Configurable<float> dcapostopv{"dcapostopv", .1, "DCA Pos To PV"};
  Configurable<int> mincrossedrows{"mincrossedrows", 70, "min crossed rows"};
  Configurable<int> tpcrefit{"tpcrefit", 0, "demand TPC refit"};

  Produces<aod::V0GoodPosTracks> v0GoodPosTracks;
  Produces<aod::V0GoodNegTracks> v0GoodNegTracks;

  void process(aod::Collision const& collision,
      MyTracksIU const& tracks)
  {
    for (auto& t0 : tracks) {
      registry.fill(HIST("hGoodTrackCount"), 0.5);
      registry.fill(HIST("hCrossedRows"), t0.tpcNClsCrossedRows());
      if (tpcrefit) {
        if (!(t0.trackType() & o2::aod::track::TPCrefit)) {
          continue; // TPC refit
        }
      }
      registry.fill(HIST("hGoodTrackCount"), 1.5);
      if (t0.tpcNClsCrossedRows() < mincrossedrows) {
        continue;
      }
      registry.fill(HIST("hGoodTrackCount"), 2.5);
      if (t0.signed1Pt() > 0.0f) {
        /*if (fabs(t0.dcaXY()) < dcapostopv) {
          continue;
          }*/
        v0GoodPosTracks(t0.globalIndex(), t0.collisionId(), t0.dcaXY());
        registry.fill(HIST("hGoodPosTrackCount"), 0.5);
        registry.fill(HIST("hGoodTrackCount"), 3.5);
      }
      if (t0.signed1Pt() < 0.0f) {
        /*if (fabs(t0.dcaXY()) < dcanegtopv) {
          continue;
          }*/
        v0GoodNegTracks(t0.globalIndex(), t0.collisionId(), -t0.dcaXY());
        registry.fill(HIST("hGoodNegTrackCount"), 0.5);
        registry.fill(HIST("hGoodTrackCount"), 3.5);
      }
    }
  }
};

struct hypertritonfinder {
  // Configurables
  Configurable<double> d_UseAbsDCA{"d_UseAbsDCA", kTRUE, "Use Abs DCAs"};
  Configurable<double> d_bz_input{"d_bz", -999, "bz field, -999 is automatic"};

  // Selection criteria
  Configurable<double> v0cospa{"v0cospa", 0.995, "V0 CosPA"}; // double -> N.B. dcos(x)/dx = 0 at x=0)
  Configurable<float> dcav0dau{"dcav0dau", 1.0, "DCA V0 Daughters"};
  //Configurable<float> v0radius{"v0radius", 5.0, "v0radius"};

  Produces<aod::StoredV0Datas> v0data;
  Produces<aod::V0s> v0;
  Produces<aod::V0DataLink> v0datalink;

  HistogramRegistry registry{
    "registry",
      {
        {"hCandPerEvent", "hCandPerEvent", {HistType::kTH1F, {{1000, 0.0f, 1000.0f}}}},
        {"hV0CutCounter", "hV0CutCounter", {HistType::kTH1F, {{5, 0.0f, 5.0f}}}},
      },
  };
  //------------------copy from lamdakzerobuilder---------------------

  Service<o2::ccdb::BasicCCDBManager> ccdb;
  Configurable<int> useMatCorrType{"useMatCorrType", 0, "0: none, 1: TGeo, 2: LUT"};
  int mRunNumber;
  float d_bz;
  float maxSnp;  //max sine phi for propagation
  float maxStep; //max step size (cm) for propagation
  void init(InitContext& context)
  {
    // using namespace analysis::lambdakzerobuilder;
    mRunNumber = 0;
    d_bz = 0;
    maxSnp = 0.85f;  //could be changed later
    maxStep = 2.00f; //could be changed later

    ccdb->setURL("https://alice-ccdb.cern.ch");
    ccdb->setCaching(true);
    ccdb->setLocalObjectValidityChecking();

    auto lut = o2::base::MatLayerCylSet::rectifyPtrFromFile(ccdb->get<o2::base::MatLayerCylSet>("GLO/Param/MatLUT"));

    if (!o2::base::GeometryManager::isGeometryLoaded()) {
      ccdb->get<TGeoManager>("GLO/Config/Geometry");
      /* it seems this is needed at this level for the material LUT to work properly */
      /* but what happens if the run changes while doing the processing?             */
      constexpr long run3grp_timestamp = (1619781650000 + 1619781529000) / 2;

      o2::parameters::GRPObject* grpo = ccdb->getForTimeStamp<o2::parameters::GRPObject>("GLO/GRP/GRP", run3grp_timestamp);
      o2::base::Propagator::initFieldFromGRP(grpo);
      o2::base::Propagator::Instance()->setMatLUT(lut);
    }

    registry.get<TH1>(HIST("hV0CutCounter"))->GetXaxis()->SetBinLabel(1, "DiffCol");
    registry.get<TH1>(HIST("hV0CutCounter"))->GetXaxis()->SetBinLabel(2, "hasSV");
    registry.get<TH1>(HIST("hV0CutCounter"))->GetXaxis()->SetBinLabel(3, "hasSV2");
    registry.get<TH1>(HIST("hV0CutCounter"))->GetXaxis()->SetBinLabel(4, "Dcav0Dau");
    registry.get<TH1>(HIST("hV0CutCounter"))->GetXaxis()->SetBinLabel(5, "CosPA");
  }

  float getMagneticField(uint64_t timestamp)
  {
    // TODO done only once (and not per run). Will be replaced by CCDBConfigurable
    static o2::parameters::GRPObject* grpo = nullptr;
    if (grpo == nullptr) {
      grpo = ccdb->getForTimeStamp<o2::parameters::GRPObject>("GLO/GRP/GRP", timestamp);
      if (grpo == nullptr) {
        LOGF(fatal, "GRP object not found for timestamp %llu", timestamp);
        return 0;
      }
      LOGF(info, "Retrieved GRP for timestamp %llu with magnetic field of %d kG", timestamp, grpo->getNominalL3Field());
    }
    float output = grpo->getNominalL3Field();
    return output;
  }

  void CheckAndUpdate(Int_t lRunNumber, uint64_t lTimeStamp)
  {
    if (lRunNumber != mRunNumber) {
      if (d_bz_input < -990) {
        // Fetch magnetic field from ccdb for current collision
        d_bz = getMagneticField(lTimeStamp);
      } else {
        d_bz = d_bz_input;
      }
      mRunNumber = lRunNumber;
    }
  }
  //------------------------------------------------------------------

  void process(aod::Collision const& collision, MyTracksIU const& tracks,
      aod::V0GoodPosTracks const& ptracks, aod::V0GoodNegTracks const& ntracks, aod::BCsWithTimestamps const&)
  {

    auto bc = collision.bc_as<aod::BCsWithTimestamps>();
    CheckAndUpdate(bc.runNumber(), bc.timestamp());

    // Define o2 fitter, 2-prong
    o2::vertexing::DCAFitterN<2> fitter;
    fitter.setBz(d_bz);
    fitter.setPropagateToPCA(true);
    fitter.setMaxR(200.);
    fitter.setMinParamChange(1e-3);
    fitter.setMinRelChi2Change(0.9);
    fitter.setMaxDZIni(1e9);
    fitter.setMaxChi2(1e9);
    fitter.setUseAbsDCA(d_UseAbsDCA);

    Long_t lNCand = 0;

    for (auto& t0id : ptracks) { // FIXME: turn into combination(...)
      for (auto& t1id : ntracks) {

        if (t0id.collisionId() != t1id.collisionId()) {
          continue;
        }
        registry.fill(HIST("hV0CutCounter"), 0.5);

        auto t0 = t0id.goodTrack_as<MyTracksIU>();
        auto t1 = t1id.goodTrack_as<MyTracksIU>();
        auto Track1 = getTrackParCov(t0);
        auto Track2 = getTrackParCov(t1);
        auto pTrack = getTrackParCov(t0);
        auto nTrack = getTrackParCov(t1);

        // Try to progate to dca
        int nCand = fitter.process(Track1, Track2);
        if (nCand == 0) {
          continue;
        }
        registry.fill(HIST("hV0CutCounter"), 1.5);

        //------------------copy from lamdakzerobuilder---------------------
        double finalXpos = fitter.getTrack(0).getX();
        double finalXneg = fitter.getTrack(1).getX();

        // Rotate to desired alpha
        pTrack.rotateParam(fitter.getTrack(0).getAlpha());
        nTrack.rotateParam(fitter.getTrack(1).getAlpha());

        // Retry closer to minimum with material corrections
        o2::base::Propagator::MatCorrType matCorr = o2::base::Propagator::MatCorrType::USEMatCorrNONE;
        if (useMatCorrType == 1)
          matCorr = o2::base::Propagator::MatCorrType::USEMatCorrTGeo;
        if (useMatCorrType == 2)
          matCorr = o2::base::Propagator::MatCorrType::USEMatCorrLUT;

        o2::base::Propagator::Instance()->propagateToX(pTrack, finalXpos, d_bz, maxSnp, maxStep, matCorr);
        o2::base::Propagator::Instance()->propagateToX(nTrack, finalXneg, d_bz, maxSnp, maxStep, matCorr);

        nCand = fitter.process(pTrack, nTrack);
        if (nCand == 0) {
          continue;
        }
        registry.fill(HIST("hV0CutCounter"), 2.5);

        //------------------------------------------------------------------

        const auto& vtx = fitter.getPCACandidate();
        // Fiducial: min radius
        /*auto thisv0radius = TMath::Sqrt(TMath::Power(vtx[0], 2) + TMath::Power(vtx[1], 2));
          if (thisv0radius < v0radius) {
          continue;
          }*/

        // DCA V0 daughters
        auto thisdcav0dau = fitter.getChi2AtPCACandidate();
        if (thisdcav0dau > dcav0dau) {
          continue;
        }
        registry.fill(HIST("hV0CutCounter"), 3.5);

        std::array<float, 3> pos = {0.};
        std::array<float, 3> pvec0;
        std::array<float, 3> pvec1;
        for (int i = 0; i < 3; i++) {
          pos[i] = vtx[i];
        }
        //fitter.getTrack(0).getPxPyPzGlo(pvec0);
        //fitter.getTrack(1).getPxPyPzGlo(pvec1);

        //------------------copy from lamdakzerobuilder---------------------
        pTrack.getPxPyPzGlo(pvec0);
        nTrack.getPxPyPzGlo(pvec1);
        //------------------------------------------------------------------
        /*uint32_t pTrackPID = t0.pidForTracking();
          uint32_t nTrackPID = t1.pidForTracking();
          int pTrackCharge = o2::track::pid_constants::sCharges[pTrackPID];
          int nTrackCharge = o2::track::pid_constants::sCharges[nTrackPID];
          for (int i=0; i<3; i++){
          pvec0[i] = pvec0[i] * pTrackCharge;
          pvec1[i] = pvec1[i] * nTrackCharge;
          }*/
        int pTrackCharge = 1, nTrackCharge = 1;
        if (TMath::Abs( GetTPCNSigmaHe3( 2*t0.p(), t0.tpcSignal()) ) < 5){
          pTrackCharge = 2;
        } 
        if (TMath::Abs( GetTPCNSigmaHe3( 2*t1.p(), t1.tpcSignal()) ) < 5){
          nTrackCharge = 2;
        } 
        for (int i=0; i<3; i++){
          pvec0[i] = pvec0[i] * pTrackCharge;
          pvec1[i] = pvec1[i] * nTrackCharge;
        }


        auto thisv0cospa = RecoDecay::cpa(array{collision.posX(), collision.posY(), collision.posZ()},
            array{vtx[0], vtx[1], vtx[2]}, array{pvec0[0] + pvec1[0], pvec0[1] + pvec1[1], pvec0[2] + pvec1[2]});
        if (thisv0cospa < v0cospa) {
          continue;
        }
        registry.fill(HIST("hV0CutCounter"), 4.5);

        lNCand++;
        v0(t0.collisionId(), t0.globalIndex(), t1.globalIndex());
        //there is a change in the position of "0" compared with lambdakzerofinder.cxx
        v0data(t0.globalIndex(), t1.globalIndex(), t0.collisionId(), 0,
            fitter.getTrack(0).getX(), fitter.getTrack(1).getX(),
            pos[0], pos[1], pos[2],
            pvec0[0], pvec0[1], pvec0[2],
            pvec1[0], pvec1[1], pvec1[2],
            fitter.getChi2AtPCACandidate(),
            t0id.dcaXY(), t1id.dcaXY());
        v0datalink(v0data.lastIndex());
      }
    }
    registry.fill(HIST("hCandPerEvent"), lNCand);
  }
};

/*struct hypertritonfinderQA {
  // Basic checks
  // Selection criteria
  Configurable<double> v0cospa{"v0cospa", 0.998, "V0 CosPA"}; // double -> N.B. dcos(x)/dx = 0 at x=0)
  Configurable<float> dcav0dau{"dcav0dau", .6, "DCA V0 Daughters"};
  Configurable<float> dcanegtopv{"dcanegtopv", .1, "DCA Neg To PV"};
  Configurable<float> dcapostopv{"dcapostopv", .1, "DCA Pos To PV"};
  Configurable<float> v0radius{"v0radius", 5.0, "v0radius"};

  HistogramRegistry registry{
    "registry",
      {
        {"hCandPerEvent", "hCandPerEvent", {HistType::kTH1F, {{1000, 0.0f, 1000.0f}}}},

        {"hV0Radius", "hV0Radius", {HistType::kTH1F, {{1000, 0.0f, 100.0f}}}},
        {"hV0CosPA", "hV0CosPA", {HistType::kTH1F, {{1000, 0.95f, 1.0f}}}},
        {"hDCAPosToPV", "hDCAPosToPV", {HistType::kTH1F, {{1000, 0.0f, 10.0f}}}},
        {"hDCANegToPV", "hDCANegToPV", {HistType::kTH1F, {{1000, 0.0f, 10.0f}}}},
        {"hDCAV0Dau", "hDCAV0Dau", {HistType::kTH1F, {{1000, 0.0f, 10.0f}}}},

        {"h3dMassHypertriton", "h3dMassHypertriton", {HistType::kTH3F, {{20, 0.0f, 100.0f, "Cent (%)"}, {200, 0.0f, 10.0f, "#it{p}_{T} (GeV/c)"}, {40, 2.95f, 3.05f, "Inv. Mass (GeV/c^{2})"}}}},
        {"h3dMassAntiHypertriton", "h3dMassAntiHypertriton", {HistType::kTH3F, {{20, 0.0f, 100.0f, "Cent (%)"}, {200, 0.0f, 10.0f, "#it{p}_{T} (GeV/c)"}, {40, 2.95f, 3.05f, "Inv. Mass (GeV/c^{2})"}}}},
      },
  };

  //Filter preFilterV0 = nabs(aod::v0data::dcapostopv) > dcapostopv&& nabs(aod::v0data::dcanegtopv) > dcanegtopv&& aod::v0data::dcaV0daughters < dcav0dau;

  /// Connect to V0Data: newly indexed, note: V0Datas table incompatible with standard V0 table!
  void processRun3(soa::Join<aod::Collisions, aod::EvSels, aod::CentFV0As>::iterator const& collision,
      //soa::Filtered<aod::V0Datas> const& fullV0s)
      aod::V0Datas const& fullV0s)
  {
    if (!collision.sel8()) {
      return;
    }

    Long_t lNCand = 0;
    for (auto& v0 : fullV0s) {
      if (v0.v0radius() > v0radius && v0.v0cosPA(collision.posX(), collision.posY(), collision.posZ()) > v0cospa) {
        registry.fill(HIST("hV0Radius"), v0.v0radius());
        registry.fill(HIST("hV0CosPA"), v0.v0cosPA(collision.posX(), collision.posY(), collision.posZ()));
        registry.fill(HIST("hDCAPosToPV"), v0.dcapostopv());
        registry.fill(HIST("hDCANegToPV"), v0.dcanegtopv());
        registry.fill(HIST("hDCAV0Dau"), v0.dcaV0daughters());

        if (TMath::Abs(v0.yHypertriton()) < 0.5) {
          registry.fill(HIST("h3dMassHypertriton"), collision.centFV0A(), v0.pt(), v0.mLambda());
          registry.fill(HIST("h3dMassAntiHypertriton"), collision.centFV0A(), v0.pt(), v0.mAntiLambda());
        }
        lNCand++;
      }
    }
    registry.fill(HIST("hCandPerEvent"), lNCand);
  }
  PROCESS_SWITCH(hypertritonfinderQA, processRun3, "Process Run 3 data", true);

  void processRun2(soa::Join<aod::Collisions, aod::EvSels, aod::CentRun2V0Ms>::iterator const& collision,
      //soa::Filtered<aod::V0Datas> const& fullV0s)
      aod::V0Datas const& fullV0s)
  {
    if (!collision.alias()[kINT7]) {
      return;
    }
    if (!collision.sel7()) {
      return;
    }

    Long_t lNCand = 0;
    for (auto& v0 : fullV0s) {
      if (v0.v0radius() > v0radius && v0.v0cosPA(collision.posX(), collision.posY(), collision.posZ()) > v0cospa) {
        registry.fill(HIST("hV0Radius"), v0.v0radius());
        registry.fill(HIST("hV0CosPA"), v0.v0cosPA(collision.posX(), collision.posY(), collision.posZ()));
        registry.fill(HIST("hDCAPosToPV"), v0.dcapostopv());
        registry.fill(HIST("hDCANegToPV"), v0.dcanegtopv());
        registry.fill(HIST("hDCAV0Dau"), v0.dcaV0daughters());

        if (TMath::Abs(v0.yHypertriton()) < 0.5) {
          registry.fill(HIST("h3dMassHypertriton"), collision.centRun2V0M(), v0.pt(), v0.mHypertriton());
          registry.fill(HIST("h3dMassAntiHypertriton"), collision.centRun2V0M(), v0.pt(), v0.mAntiHypertriton());
        }
        lNCand++;
      }
    }
    registry.fill(HIST("hCandPerEvent"), lNCand);
  }
  PROCESS_SWITCH(hypertritonfinderQA, processRun2, "Process Run 2 data", false);

};*/

/// Extends the v0data table with expression columns
struct hypertritoninitializer {
  Spawns<aod::V0Datas> v0datas;
  void init(InitContext const&) {}
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<hypertritonprefilter>(cfgc, TaskName{"lf-hypertritonprefilter"}),
      adaptAnalysisTask<hypertritonfinder>(cfgc, TaskName{"lf-hypertritonfinder"}),
      //adaptAnalysisTask<hypertritonfinderQA>(cfgc, TaskName{"lf-hypertritonfinderQA"}),
      adaptAnalysisTask<hypertritoninitializer>(cfgc, TaskName{"lf-hypertritoninitializer"})};
}
