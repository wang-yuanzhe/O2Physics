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
// ========================
//
// Simple check for injected hyper-helium4sigma (H4S)
//

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "ReconstructionDataFormats/Track.h"
#include "Common/Core/RecoDecay.h"
#include "Common/Core/trackUtilities.h"
#include "Common/Core/TrackSelection.h"
#include "Common/DataModel/TrackSelectionTables.h"
#include "Common/TableProducer/PID/pidTOFBase.h"
#include "Common/DataModel/EventSelection.h"
#include "Common/DataModel/Centrality.h"
#include "Common/DataModel/PIDResponse.h"
#include "CommonConstants/PhysicsConstants.h"
#include "CCDB/BasicCCDBManager.h"

#include "PWGLF/DataModel/pidTOFGeneric.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;
using ColwithEvTimes = o2::soa::Join<aod::Collisions, o2::aod::McCollisionLabels, aod::EvSels, aod::EvTimeTOFFT0>;
using FullTracksExtIU = soa::Join<aod::TracksIU, aod::TracksExtra, aod::TracksCovIU, aod::TracksDCA, aod::pidTPCFullPr, aod::pidTPCFullAl, aod::pidTPCFullTr, aod::pidTPCFullPi, aod::pidTPCFullDe, aod::TOFEvTime, aod::TOFSignal, aod::EvTimeTOFFT0ForTrack>;
using MCLabeledTracksIU = soa::Join<FullTracksExtIU, aod::McTrackLabels>;

//-------------------------------Check the decay channel of H4S-------------------------------

enum dChannel {
  k2body = 0, // helium4, pion0
  k3body_p,   // triton, proton, pion0
  k3body_n,   // triton, neutron, pion+
  kNDChannel
};

template <class TMCTrackTo, typename TMCParticle>
dChannel GetDChannelH4S(TMCParticle const& particle)
{
  if (std::abs(particle.pdgCode()) != 1110020040) {
    return kNDChannel;
  }
  bool haveAlpha = false, haveTriton = false, haveProton = false, haveNeuteron = false;
  bool haveAntiAlpha = false, haveAntiTriton = false, haveAntiProton = false, haveAntiNeuteron = false;
  bool havePionPlus = false, havePionMinus = false, havePion0 = false;
  for (auto& mcDaughter : particle.template daughters_as<TMCTrackTo>()) {
    if (mcDaughter.pdgCode() == 1000020040)
      haveAlpha = true;
    if (mcDaughter.pdgCode() == -1000020040)
      haveAntiAlpha = true;
    if (mcDaughter.pdgCode() == 1000010030)
      haveTriton = true;
    if (mcDaughter.pdgCode() == -1000010030)
      haveAntiTriton = true;
    if (mcDaughter.pdgCode() == 2212)
      haveProton = true;
    if (mcDaughter.pdgCode() == -2212)
      haveAntiProton = true;
    if (mcDaughter.pdgCode() == 2112)
      haveNeuteron = true;
    if (mcDaughter.pdgCode() == -2112)
      haveAntiNeuteron = true;
    if (mcDaughter.pdgCode() == 211)
      havePionPlus = true;
    if (mcDaughter.pdgCode() == -211)
      havePionMinus = true;
    if (mcDaughter.pdgCode() == 111)
      havePion0 = true;
  }

  if ((haveAlpha && havePion0) || (haveAntiAlpha && havePion0)) {
    return k2body;
  } else if ((haveTriton && haveProton && havePion0) || (haveAntiTriton && haveAntiProton && havePion0)) {
    return k3body_p;
  } else if ((haveTriton && haveNeuteron && havePionPlus) || (haveAntiTriton && haveAntiNeuteron && havePionMinus)) {
    return k3body_n;
  }

  return kNDChannel;
}
//--------------------------------------------------------------

// check the performance of mcparticle
struct hyperhelium4sigmaMcParticleCheck {
  // Basic checks
  HistogramRegistry registry{
    "registry",
    {
      {"hMcCollCounter", "hMcCollCounter", {HistType::kTH1F, {{2, 0.0f, 2.0f}}}},
      {"hMcHyperHelium4SigmaCounter", "hMcHyperHelium4SigmaCounter", {HistType::kTH1F, {{6, 0.0f, 6.0f}}}},
      {"hMcRecoInvMass", "hMcRecoInvMass", {HistType::kTH1F, {{100, 3.85, 4.15f}}}},
    },
  };

  o2::pid::tof::TOFResoParamsV2 mRespParamsV2;

  void init(InitContext&)
  {
    registry.get<TH1>(HIST("hMcCollCounter"))->GetXaxis()->SetBinLabel(1, "Total Counter");
    registry.get<TH1>(HIST("hMcCollCounter"))->GetXaxis()->SetBinLabel(2, "Reconstructed");

    registry.get<TH1>(HIST("hMcHyperHelium4SigmaCounter"))->GetXaxis()->SetBinLabel(1, "H4S All");
    registry.get<TH1>(HIST("hMcHyperHelium4SigmaCounter"))->GetXaxis()->SetBinLabel(2, "Matter");
    registry.get<TH1>(HIST("hMcHyperHelium4SigmaCounter"))->GetXaxis()->SetBinLabel(3, "AntiMatter");
    registry.get<TH1>(HIST("hMcHyperHelium4SigmaCounter"))->GetXaxis()->SetBinLabel(4, "He4, #pi^{0}");
    registry.get<TH1>(HIST("hMcHyperHelium4SigmaCounter"))->GetXaxis()->SetBinLabel(5, "Triton, Proton, #pi^{0}");
    registry.get<TH1>(HIST("hMcHyperHelium4SigmaCounter"))->GetXaxis()->SetBinLabel(6, "Triton, Neutron, #pi^{+}");
  }

  Configurable<bool> mc_event_selection{"mc_event_selection", false, "mc event selection"};
  Configurable<bool> event_posZ_selection{"event_posZ_selection", false, "event selection count post poZ cut"};

  Preslice<aod::McParticles> permcCollision = o2::aod::mcparticle::mcCollisionId;

  std::vector<int64_t> mcPartIndices;
  template <typename TTrackTable>
  void SetTrackIDForMC(aod::McParticles const& particlesMC, TTrackTable const& tracks)
  {
    mcPartIndices.clear();
    mcPartIndices.resize(particlesMC.size());
    std::fill(mcPartIndices.begin(), mcPartIndices.end(), -1);
    for (auto& track : tracks) {
      if (track.has_mcParticle()) {
        auto mcparticle = track.template mcParticle_as<aod::McParticles>();
        if (mcPartIndices[mcparticle.globalIndex()] == -1) {
          mcPartIndices[mcparticle.globalIndex()] = track.globalIndex();
        } else {
          auto candTrack = tracks.rawIteratorAt(mcPartIndices[mcparticle.globalIndex()]);
          // Use the track which has innest information (also best quality?
          if (track.x() < candTrack.x()) {
            mcPartIndices[mcparticle.globalIndex()] = track.globalIndex();
          }
        }
      }
    }
  }

  void process(aod::McCollisions const& mcCollisions, aod::McParticles const& particlesMC, o2::soa::Join<o2::aod::Collisions, o2::aod::McCollisionLabels, o2::aod::EvSels> const& collisions, MCLabeledTracksIU const& tracks)
  {
    SetTrackIDForMC(particlesMC, tracks);
    std::vector<int64_t> SelectedEvents(collisions.size());
    LOG(info) << "CollisionsSize: " << SelectedEvents.size();
    int nevts = 0;
    for (const auto& collision : collisions) {
      if (mc_event_selection && (!collision.selection_bit(aod::evsel::kIsTriggerTVX) || !collision.selection_bit(aod::evsel::kNoTimeFrameBorder))) {
        continue;
      }
      if (event_posZ_selection && abs(collision.posZ()) > 10.f) { // 10cm
        continue;
      }
      SelectedEvents[nevts++] = collision.mcCollision_as<aod::McCollisions>().globalIndex();
      // LOG(info) << "SelectedEvents: " << collision.mcCollision_as<aod::McCollisions>().globalIndex();
    }
    SelectedEvents.resize(nevts);
    LOG(info) << "SelectedEvents size: " << SelectedEvents.size();

    for (auto mcCollision : mcCollisions) {
      registry.fill(HIST("hMcCollCounter"), 0.5);
      const auto evtReconstructedAndSelected = std::find(SelectedEvents.begin(), SelectedEvents.end(), mcCollision.globalIndex()) != SelectedEvents.end();
      if (evtReconstructedAndSelected) { // Check that the event is reconstructed and that the reconstructed events pass the selection
        registry.fill(HIST("hMcCollCounter"), 1.5);
      } else {
        // continue;
      }

      const auto& dparticlesMC = particlesMC.sliceBy(permcCollision, mcCollision.globalIndex());

      for (auto& mcparticle : dparticlesMC) {

        if (mcparticle.pdgCode() == 1110020040) {
          registry.fill(HIST("hMcHyperHelium4SigmaCounter"), 1.5);
        } else if (mcparticle.pdgCode() == -1110020040) {
          registry.fill(HIST("hMcHyperHelium4SigmaCounter"), 2.5);
        } else {
          continue;
        }

        registry.fill(HIST("hMcHyperHelium4SigmaCounter"), 0.5);

        double decayPos[3] = {-999, -999, -999};
        double dauHelium4Mom[3] = {-999, -999, -999};
        double dauTritonMom[3] = {-999, -999, -999};
        double dauProtonMom[3] = {-999, -999, -999};
        double dauNeuteronMom[3] = {-999, -999, -999};
        double dauChargedPionMom[3] = {-999, -999, -999};
        double dauPion0Mom[3] = {-999, -999, -999};
        double MClifetime = 999;
        auto dChannel = GetDChannelH4S<aod::McParticles>(mcparticle);
        if (dChannel == kNDChannel) {
          continue;
        }
        for (auto& mcparticleDaughter : mcparticle.daughters_as<aod::McParticles>()) {
          if (std::abs(mcparticleDaughter.pdgCode()) == 1000020040) {
            dauHelium4Mom[0] = mcparticleDaughter.px();
            dauHelium4Mom[1] = mcparticleDaughter.py();
            dauHelium4Mom[2] = mcparticleDaughter.pz();
          } else if (std::abs(mcparticleDaughter.pdgCode()) == 1000010030) {
            dauTritonMom[0] = mcparticleDaughter.px();
            dauTritonMom[1] = mcparticleDaughter.py();
            dauTritonMom[2] = mcparticleDaughter.pz();
          } else if (std::abs(mcparticleDaughter.pdgCode()) == 2212) {
            dauProtonMom[0] = mcparticleDaughter.px();
            dauProtonMom[1] = mcparticleDaughter.py();
            dauProtonMom[2] = mcparticleDaughter.pz();
          } else if (std::abs(mcparticleDaughter.pdgCode()) == 2112) {
            dauNeuteronMom[0] = mcparticleDaughter.px();
            dauNeuteronMom[1] = mcparticleDaughter.py();
            dauNeuteronMom[2] = mcparticleDaughter.pz();
          } else if (std::abs(mcparticleDaughter.pdgCode()) == 211) {
            dauChargedPionMom[0] = mcparticleDaughter.px();
            dauChargedPionMom[1] = mcparticleDaughter.py();
            dauChargedPionMom[2] = mcparticleDaughter.pz();
          } else if (mcparticleDaughter.pdgCode() == 111) {
            dauPion0Mom[0] = mcparticleDaughter.px();
            dauPion0Mom[1] = mcparticleDaughter.py();
            dauPion0Mom[2] = mcparticleDaughter.pz();
          }
        }

        if (dChannel == k2body) {
          registry.fill(HIST("hMcHyperHelium4SigmaCounter"), 3.5);
          double hyperHelium4SigmaMCMass = RecoDecay::m(array{array{dauHelium4Mom[0], dauHelium4Mom[1], dauHelium4Mom[2]}, array{dauPion0Mom[0], dauPion0Mom[1], dauPion0Mom[2]}}, array{o2::constants::physics::MassAlpha, o2::constants::physics::MassPi0});
          registry.fill(HIST("hMcRecoInvMass"), hyperHelium4SigmaMCMass);
        } else if (dChannel == k3body_p) {
          registry.fill(HIST("hMcHyperHelium4SigmaCounter"), 4.5);
          double hyperHelium4SigmaMCMass = RecoDecay::m(array{array{dauTritonMom[0], dauTritonMom[1], dauTritonMom[2]}, array{dauProtonMom[0], dauProtonMom[1], dauProtonMom[2]}, array{dauPion0Mom[0], dauPion0Mom[1], dauPion0Mom[2]}}, array{o2::constants::physics::MassTriton, o2::constants::physics::MassProton, o2::constants::physics::MassPi0});
          registry.fill(HIST("hMcRecoInvMass"), hyperHelium4SigmaMCMass);
        } else if (dChannel == k3body_n) {
          registry.fill(HIST("hMcHyperHelium4SigmaCounter"), 5.5);
          double hyperHelium4SigmaMCMass = RecoDecay::m(array{array{dauTritonMom[0], dauTritonMom[1], dauTritonMom[2]}, array{dauNeuteronMom[0], dauNeuteronMom[1], dauNeuteronMom[2]}, array{dauChargedPionMom[0], dauChargedPionMom[1], dauChargedPionMom[2]}}, array{o2::constants::physics::MassTriton, o2::constants::physics::MassNeutron, o2::constants::physics::MassPionCharged});
          registry.fill(HIST("hMcRecoInvMass"), hyperHelium4SigmaMCMass);
        }
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<hyperhelium4sigmaMcParticleCheck>(cfgc),
  };
}
