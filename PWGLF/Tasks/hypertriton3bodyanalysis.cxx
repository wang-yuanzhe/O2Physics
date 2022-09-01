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
// Example V0 analysis task
// ========================
//
// This code loops over a V0Data table and produces some
// standard analysis output. It requires either
// the hypertritonfinder or the hypertritonproducer tasks
// to have been executed in the workflow (before).
//
//    Comments, questions, complaints, suggestions?
//    Please write to:
//    david.dobrigkeit.chinellato@cern.ch
//
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "ReconstructionDataFormats/Track.h"
#include "Common/Core/RecoDecay.h"
#include "Common/Core/trackUtilities.h"
#include "Common/DataModel/StrangenessTables.h"
#include "Common/Core/TrackSelection.h"
#include "Common/DataModel/TrackSelectionTables.h"
#include "Common/DataModel/EventSelection.h"
#include "Common/DataModel/Centrality.h"
#include "Common/DataModel/PIDResponse.h"

#include "../DataModel/Vtx3BodyTables.h"

#include <TFile.h>
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

//using MyTracks = soa::Join<aod::Tracks, aod::TracksExtra, aod::pidTPCPi, aod::pidTPCDe, aod::pidTPCTr, aod::pidTPCKa, aod::pidTPCPr>;
using MyTracks = soa::Join<aod::Tracks, aod::TracksExtra, aod::pidTPCFullPi, aod::pidTPCFullDe, aod::pidTPCFullTr, aod::pidTPCFullKa, aod::pidTPCFullPr>;

struct hypertriton3bodyQa {
  //Basic checks
  HistogramRegistry registry{
    "registry",
      {
        {"hVtxRadius", "hVtxRadius", {HistType::kTH1F, {{1000, 0.0f, 100.0f, "cm"}}}},
        {"hVtxCosPA", "hVtxCosPA", {HistType::kTH1F, {{1000, 0.95f, 1.0f}}}},
        {"hDCATrack0ToPV", "hDCAPosToPV", {HistType::kTH1F, {{1000, -10.0f, 10.0f, "cm"}}}},
        {"hDCATrack1ToPV", "hDCANegToPV", {HistType::kTH1F, {{1000, 10.0f, 10.0f, "cm"}}}},
        {"hDCATrack2ToPV", "hDCANegToPV", {HistType::kTH1F, {{1000, 10.0f, 10.0f, "cm"}}}},
        {"hDCAVtxDau", "hDCAVtxDau", {HistType::kTH1F, {{1000, 0.0f, 10.0f, "cm^{2}"}}}},
        {"hVtxPt", "hVtxPt", {HistType::kTH1F, {{200, 0.0f, 10.0f, "p_{T}"}}}},
        {"hTrack0Pt", "hTrack0Pt", {HistType::kTH1F, {{200, 0.0f, 10.0f, "p_{T}"}}}},
        {"hTrack1Pt", "hTrack1Pt", {HistType::kTH1F, {{200, 0.0f, 10.0f, "p_{T}"}}}},
        {"hTrack2Pt", "hTrack2Pt", {HistType::kTH1F, {{200, 0.0f, 10.0f, "p_{T}"}}}},
      },
  };
  void init(InitContext const&)
  {
    AxisSpec massAxis = {120, 2.9f, 3.2f, "Inv. Mass (GeV/c^{2})"};

    registry.add("hMassHypertriton", "hMassHypertriton", {HistType::kTH1F, {massAxis}});
    registry.add("hMassAntiHypertriton", "hMassAntiHypertriton", {HistType::kTH1F, {massAxis}});
  }
  void process(aod::Collision const& collision, aod::Vtx3BodyDatas const& vtx3BodyDatas, MyTracks const& tracks)
  {

    for (auto& vtx : vtx3BodyDatas) {
      registry.fill(HIST("hVtxRadius"), vtx.vtxradius());
      registry.fill(HIST("hVtxCosPA"), vtx.vtxcosPA(collision.posX(), collision.posY(), collision.posZ()));
      registry.fill(HIST("hDCATrack0ToPV"), vtx.dcatrack0topv());
      registry.fill(HIST("hDCATrack1ToPV"), vtx.dcatrack1topv());
      registry.fill(HIST("hDCATrack2ToPV"), vtx.dcatrack2topv());
      registry.fill(HIST("hDCAVtxDau"), vtx.dcaVtxdaughters());
      registry.fill(HIST("hVtxPt"), vtx.pt());
      registry.fill(HIST("hTrack0Pt"), vtx.track0pt());
      registry.fill(HIST("hTrack1Pt"), vtx.track1pt());
      registry.fill(HIST("hTrack2Pt"), vtx.track2pt());
      registry.fill(HIST("hMassHypertriton"), vtx.mHypertriton());
      registry.fill(HIST("hMassAntiHypertriton"), vtx.mAntiHypertriton());
    }
  }
};

struct hypertriton3bodyAnalysis {

  HistogramRegistry registry{
    "registry",
      {
        {"hSelectedEventCounter", "hSelectedEventCounter", {HistType::kTH1F, {{2, 0.0f, 2.0f}}}},
        {"hSelectedCandidatesCounter", "hSelectedCandidatesCounter", {HistType::kTH1F, {{9, 0.0f, 9.0f}}}},
        {"hTestCounter", "hTestCounter", {HistType::kTH1F, {{9, 0.0f, 9.0f}}}},
        {"hMassHypertriton", "hMassHypertriton", {HistType::kTH1F, {{40, 2.95f, 3.05f}}}},
        {"hMassAntiHypertriton", "hMassAntiHypertriton", {HistType::kTH1F, {{40, 2.95f, 3.05f}}}},
        {"hMassHypertritonTotal", "hMassHypertritonTotal", {HistType::kTH1F, {{120, 2.9f, 3.2f}}}},
        {"hNSigmaHelium3", "hNSigmaHelium3", {HistType::kTH1F, {{240, -6.0f, 6.0f}}}},
        {"hNSigmaPion", "hNSigmaPion", {HistType::kTH1F, {{240, -6.0f, 6.0f}}}},
        {"hNSigmaTriton", "hNSigmaTriton", {HistType::kTH1F, {{240, -6.0f, 6.0f}}}},
        {"hNSigmaKaon", "hNSigmaKaon", {HistType::kTH1F, {{240, -6.0f, 6.0f}}}},
        {"hNSigmaProton", "hNSigmaProton", {HistType::kTH1F, {{240, -6.0f, 6.0f}}}},
        {"hPtProton", "hPtProton", {HistType::kTH1F, {{200, 0.0f, 10.0f}}}},
        {"hPtAntiPion", "hPtAntiPion", {HistType::kTH1F, {{200, 0.0f, 10.0f}}}},
        {"hPtDeuteron", "hPtDeuteron", {HistType::kTH1F, {{200, 0.0f, 10.0f}}}},
        {"hPtAntiProton", "hPtAntiProton", {HistType::kTH1F, {{200, 0.0f, 10.0f}}}},
        {"hPtPion", "hPtPion", {HistType::kTH1F, {{200, 0.0f, 10.0f}}}},
        {"hPtAntiDeuteron", "hPtAntiDeuteron", {HistType::kTH1F, {{200, 0.0f, 10.0f}}}},
        {"h3dMassHypertriton", "h3dMassHypertriton", {HistType::kTH3F, {{20, 0.0f, 100.0f, "Cent (%)"}, {200, 0.0f, 10.0f, "#it{p}_{T} (GeV/c)"}, {40, 2.95f, 3.05f, "Inv. Mass (GeV/c^{2})"}}}},
        {"h3dMassAntiHypertriton", "h3dMassAntiHypertriton", {HistType::kTH3F, {{20, 0.0f, 100.0f, "Cent (%)"}, {200, 0.0f, 10.0f, "#it{p}_{T} (GeV/c)"}, {40, 2.95f, 3.05f, "Inv. Mass (GeV/c^{2})"}}}},
        {"h3dTotalHypertriton", "h3dTotalHypertriton", {HistType::kTH3F, {{50, 0, 50, "ct(cm)"}, {200, 0.0f, 10.0f, "#it{p}_{T} (GeV/c)"}, {40, 2.95f, 3.05f, "Inv. Mass (GeV/c^{2})"}}}},
      },
  };

  Configurable<int> saveDcaHist{"saveDcaHist", 0, "saveDcaHist"};

  ConfigurableAxis dcaBinning{"dca-binning", {200, 0.0f, 1.0f}, ""};
  ConfigurableAxis ptBinning{"pt-binning", {200, 0.0f, 10.0f}, ""};

  void init(InitContext const&)
  {
    AxisSpec dcaAxis = {dcaBinning, "DCA (cm)"};
    AxisSpec ptAxis = {ptBinning, "#it{p}_{T} (GeV/c)"};
    AxisSpec massAxisHypertriton = {40, 2.95f, 3.05f, "Inv. Mass (GeV/c^{2})"};

    if (saveDcaHist==1){
      registry.add("h3dMassHypertritonDca", "h3dMassHypertritonDca", {HistType::kTH3F, {dcaAxis, ptAxis, massAxisHypertriton}});
      registry.add("h3dMassAntiHypertritonDca", "h3dMassAntiHypertritonDca", {HistType::kTH3F, {dcaAxis, ptAxis, massAxisHypertriton}});
    }

    registry.get<TH1>(HIST("hSelectedCandidatesCounter"))->GetXaxis()->SetBinLabel(1, "Readin");
    registry.get<TH1>(HIST("hSelectedCandidatesCounter"))->GetXaxis()->SetBinLabel(2, "VtxCosPA");
    registry.get<TH1>(HIST("hSelectedCandidatesCounter"))->GetXaxis()->SetBinLabel(3, "TrackEta");
    registry.get<TH1>(HIST("hSelectedCandidatesCounter"))->GetXaxis()->SetBinLabel(4, "MomRapidity");
    registry.get<TH1>(HIST("hSelectedCandidatesCounter"))->GetXaxis()->SetBinLabel(5, "Lifetime");
    registry.get<TH1>(HIST("hSelectedCandidatesCounter"))->GetXaxis()->SetBinLabel(6, "DcaV0Dau");
    registry.get<TH1>(HIST("hSelectedCandidatesCounter"))->GetXaxis()->SetBinLabel(7, "TPCPID");
    registry.get<TH1>(HIST("hSelectedCandidatesCounter"))->GetXaxis()->SetBinLabel(8, "PtCut");
    registry.get<TH1>(HIST("hSelectedCandidatesCounter"))->GetXaxis()->SetBinLabel(9, "PionDcatoPV");

  }

  //Selection criteria
  Configurable<double> vtxcospa{"vtxcospa", 0.995, "Vtx CosPA"}; //double -> N.B. dcos(x)/dx = 0 at x=0)
  Configurable<float> dcavtxdau{"dcavtxdau", 1.0, "DCA Vtx Daughters"};//loose cut
  Configurable<float> dcapiontopv{"dcapiontopv", .1, "DCA Pion To PV"};
  Configurable<float> vtxradius{"vtxradius", 5.0, "vtxdadius"};
  Configurable<float> etacut{"etacut", 0.9, "etacut"};
  Configurable<float> rapidity{"rapidity", 0.8, "rapidity"};
  Configurable<float> TpcPidNsigmaCut{"TpcPidNsigmaCut", 5, "TpcPidNsigmaCut"};
  Configurable<bool> eventSelection{"eventSelection", true, "event selection"};
  Configurable<float> lifetimecut{"lifetimecut", 40., "lifetimecut"}; //ct

  //Filter dcaFilterV0 = aod::vtx.ata::dcaV0daughters < dcavtx.au;

  void process(soa::Join<aod::Collisions, aod::EvSels>::iterator const& collision, aod::Vtx3BodyDatas const& vtx3BodyDatas, MyTracks const& tracks)
  {
    registry.fill(HIST("hSelectedEventCounter"), 0.5);
    /*if (eventSelection && !collision.sel8()) {
      return;
      }
      registry.fill(HIST("hSelectedEventCounter"), 1.5);*/

    for (auto& vtx : vtx3BodyDatas) {
      //FIXME: could not find out how to filter cosPA and radius variables (dynamic columns)
      registry.fill(HIST("hSelectedCandidatesCounter"), 0.5);
      if (vtx.vtxcosPA(collision.posX(), collision.posY(), collision.posZ()) < vtxcospa) {
        continue;
      }
      registry.fill(HIST("hSelectedCandidatesCounter"), 1.5);
      if ( TMath::Abs(vtx.track0_as<MyTracks>().eta()) > etacut || TMath::Abs(vtx.track1_as<MyTracks>().eta()) > etacut || TMath::Abs(vtx.track2_as<MyTracks>().eta()) > etacut ){
        continue;
      }
      registry.fill(HIST("hSelectedCandidatesCounter"), 2.5);
      if (TMath::Abs(vtx.yHypertriton()) > rapidity) {
        continue;
      }
      registry.fill(HIST("hSelectedCandidatesCounter"), 3.5);
      double ct = vtx.distovertotmom(collision.posX(), collision.posY(), collision.posZ()) * 2.991; 
      if (ct > lifetimecut) {
        continue;
      }
      registry.fill(HIST("hSelectedCandidatesCounter"), 4.5);
      if (vtx.dcaVtxdaughters() > dcavtxdau){
        continue;
      }
      registry.fill(HIST("hSelectedCandidatesCounter"), 5.5);

      /*registry.fill(HIST("hNSigmaPion"), vtx.track0_as<MyTracks>().tpcNSigmaPi());
      registry.fill(HIST("hNSigmaPion"), vtx.track1_as<MyTracks>().tpcNSigmaPi());
      registry.fill(HIST("hNSigmaTriton"), vtx.track0_as<MyTracks>().tpcNSigmaTr());
      registry.fill(HIST("hNSigmaTriton"), vtx.track1_as<MyTracks>().tpcNSigmaTr());
      registry.fill(HIST("hNSigmaKaon"), vtx.track0_as<MyTracks>().tpcNSigmaKa());
      registry.fill(HIST("hNSigmaKaon"), vtx.track1_as<MyTracks>().tpcNSigmaKa());
      registry.fill(HIST("hNSigmaProton"), vtx.track0_as<MyTracks>().tpcNSigmaPr());
      registry.fill(HIST("hNSigmaProton"), vtx.track1_as<MyTracks>().tpcNSigmaPr());*/
      // Hypertriton
      if (TMath::Abs( vtx.track0_as<MyTracks>().tpcNSigmaPr())  < TpcPidNsigmaCut && TMath::Abs(vtx.track1_as<MyTracks>().tpcNSigmaPi()) < TpcPidNsigmaCut && TMath::Abs( vtx.track2_as<MyTracks>().tpcNSigmaDe()) < TpcPidNsigmaCut ) {
        registry.fill(HIST("hSelectedCandidatesCounter"), 6.5);

        registry.fill(HIST("hTestCounter"), 0.5);
        if(vtx.track1pt() > 0.2 && vtx.track1pt() < 1.2 ){
          registry.fill(HIST("hTestCounter"), 1.5);
          if (vtx.track0pt() > 1.8 && vtx.track0pt() < 10){
            registry.fill(HIST("hTestCounter"), 2.5);
            if (vtx.pt() > 2 && vtx.pt() < 9 ){
              registry.fill(HIST("hTestCounter"), 3.5);
            }
          }
        }

        if(/*vtx.negativept() > 0.2 && vtx.negativept() < 1.2 && vtx.positivept() > 1.8 && vtx.positivept() < 10 &&*/ vtx.pt() > 2 && vtx.pt() < 9 ){
          registry.fill(HIST("hSelectedCandidatesCounter"), 7.5);

          //if (TMath::Abs(vtx.dcanegtopv()) > dcapiontopv) {
            registry.fill(HIST("hSelectedCandidatesCounter"), 8.5);

            registry.fill(HIST("hPtProton"), vtx.track0pt());
            registry.fill(HIST("hPtPion"), vtx.track1pt());
            registry.fill(HIST("hPtDeuteron"), vtx.track2pt());
            registry.fill(HIST("hMassHypertriton"), vtx.mHypertriton());
            registry.fill(HIST("hMassHypertritonTotal"), vtx.mHypertriton());
            registry.fill(HIST("h3dMassHypertriton"), 0., vtx.pt(), vtx.mHypertriton());            //collision.centV0M() instead of 0. once available
            registry.fill(HIST("h3dTotalHypertriton"), ct, vtx.pt(), vtx.mHypertriton());
            if (saveDcaHist == 1) {
              //registry.fill(HIST("h3dMassHypertritonDca"), vtx.dcaV0daughters(), vtx.pt(), vtx.mHypertriton());
            }
          //}
        }
      }

      // AntiHypertriton
      if (TMath::Abs( vtx.track1_as<MyTracks>().tpcNSigmaPr())  < TpcPidNsigmaCut && TMath::Abs(vtx.track2_as<MyTracks>().tpcNSigmaPi()) < TpcPidNsigmaCut && TMath::Abs( vtx.track2_as<MyTracks>().tpcNSigmaDe()) < TpcPidNsigmaCut ) {

        registry.fill(HIST("hSelectedCandidatesCounter"), 6.5);

        registry.fill(HIST("hTestCounter"), 0.5);
        if(vtx.track0pt() > 0.2 && vtx.track0pt() < 1.2 ){
          registry.fill(HIST("hTestCounter"), 1.5);
          if (vtx.track1pt() > 1.8 && vtx.track1pt() < 10){
            registry.fill(HIST("hTestCounter"), 2.5);
            if (vtx.pt() > 2 && vtx.pt() < 9 ){
              registry.fill(HIST("hTestCounter"), 3.5);
            }
          }
        }

        if(/*vtx.positivept() > 0.2 && vtx.positivept() < 1.2 && vtx.negativept() > 1.8 && vtx.negativept() < 10 &&*/ vtx.pt() > 2 && vtx.pt() < 9 ){
          registry.fill(HIST("hSelectedCandidatesCounter"), 7.5);
          //if (TMath::Abs(vtx.dcapostopv()) > dcapiontopv) {
            registry.fill(HIST("hSelectedCandidatesCounter"), 8.5);

            registry.fill(HIST("hPtAntiProton"), vtx.track0pt());
            registry.fill(HIST("hPtPion"), vtx.track1pt());
            registry.fill(HIST("hPtAntiDeuteron"), vtx.track2pt());
            registry.fill(HIST("hMassAntiHypertriton"), vtx.mAntiHypertriton());
            registry.fill(HIST("hMassHypertritonTotal"), vtx.mAntiHypertriton());
            registry.fill(HIST("h3dMassAntiHypertriton"), 0., vtx.pt(), vtx.mAntiHypertriton());
            registry.fill(HIST("h3dTotalHypertriton"), ct, vtx.pt(), vtx.mAntiHypertriton());
            if (saveDcaHist == 1) {
              //registry.fill(HIST("h3dMassAntiHypertritonDca"), vtx.dcaV0daughters(), vtx.pt(), vtx.mAntiHypertriton());
            }
          //}
        }
      }

    }
  }

  //PROCESS_SWITCH(hypertriton3bodyAnalysis, processRun3, "Process Run 3 data", true);

};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<hypertriton3bodyAnalysis>(cfgc),
      adaptAnalysisTask<hypertriton3bodyQa>(cfgc),
  };
}
