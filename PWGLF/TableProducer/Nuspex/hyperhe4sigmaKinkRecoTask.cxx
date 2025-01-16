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
// Search for hyperhe4sigma kink decay topology, copied from hyperkinkRecoTask.cxx
// ==============================================================================

#include <array>

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "ReconstructionDataFormats/Track.h"
#include "Common/Core/trackUtilities.h"
#include "Common/DataModel/EventSelection.h"
#include "Common/DataModel/Multiplicity.h"
#include "Common/DataModel/Centrality.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GeometryManager.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "CCDB/BasicCCDBManager.h"

#include "Common/Core/PID/TPCPIDResponse.h"
#include "DataFormatsTPC/BetheBlochAleph.h"
#include "DCAFitter/DCAFitterN.h"

#include "PWGLF/DataModel/LFParticleIdentification.h"
#include "PWGLF/Utils/svPoolCreator.h"
#include "PWGLF/DataModel/LFHypernucleiTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;
using VBracket = o2::math_utils::Bracket<int>;
using TracksFull = soa::Join<aod::TracksIU, aod::TracksExtra, aod::TracksCovIU, aod::pidTPCFullTr, aod::pidTPCFullAl, aod::pidTOFFullTr, aod::pidTOFFullAl>;
using CollisionsFull = soa::Join<aod::Collisions, aod::EvSels, aod::CentFT0As, aod::CentFT0Cs, aod::CentFT0Ms>;
using CollisionsFullMC = soa::Join<aod::Collisions, aod::McCollisionLabels, aod::EvSels, aod::CentFT0As, aod::CentFT0Cs, aod::CentFT0Ms>;

namespace
{
constexpr std::array<float, 7> LayerRadii{2.33959f, 3.14076f, 3.91924f, 19.6213f, 24.5597f, 34.388f, 39.3329f};
constexpr double betheBlochDefault[1][6]{{-1.e32, -1.e32, -1.e32, -1.e32, -1.e32, -1.e32}};
static const std::vector<std::string> betheBlochParNames{"p0", "p1", "p2", "p3", "p4", "resolution"};
static const std::vector<std::string> particleNames{"Triton"};
std::shared_ptr<TH1> hEvents;
std::shared_ptr<TH1> hZvtx;
std::shared_ptr<TH1> hCentFT0A;
std::shared_ptr<TH1> hCentFT0C;
std::shared_ptr<TH1> hCentFT0M;
std::shared_ptr<TH1> hCentFV0A;
std::shared_ptr<TH2> hNsigmaDauSel;
std::shared_ptr<TH2> hDeDxDauSel;
std::shared_ptr<TH2> hDeDxTot;
std::shared_ptr<TH1> hIsMatterGen;
std::shared_ptr<TH1> hInvMass;
std::shared_ptr<TH1> hHyperPt;
std::shared_ptr<TH1> hCandidates;
std::shared_ptr<TH1> hTrueCandidates;
std::shared_ptr<TH1> hTracks;
std::shared_ptr<TH1> hDiffHyperPt;
std::shared_ptr<TH1> hDiffHyperP;
std::shared_ptr<TH2> hRVsDiffHyperPt;
std::shared_ptr<TH2> hRVsDiffTrueHyperPt;
std::shared_ptr<TH1> hDiffDauPx;
std::shared_ptr<TH1> hDiffDauPy;
std::shared_ptr<TH1> hDiffDauPz;
std::shared_ptr<TH1> hDiffSVx;
std::shared_ptr<TH1> hDiffSVy;
std::shared_ptr<TH1> hDiffSVz;
std::shared_ptr<TH1> hHyperITSNcls;
std::shared_ptr<TH1> hTrueHyperITSNcls;
} // namespace

struct kinkCandidate {

  float recoPtHyp() const { return std::hypot(momHyp[0], momHyp[1]); }
  float recoPhiHyp() const { return std::atan2(momHyp[1], momHyp[0]); }
  float recoEtaHyp() const { return std::asinh(momHyp[2] / recoPtHyp()); }

  float recoPtDau() const { return std::hypot(momDau[0], momDau[1]); }
  float recoPhiDau() const { return std::atan2(momDau[1], momDau[0]); }
  float recoEtaDau() const { return std::asinh(momDau[2] / recoPtDau()); }

  float genPt() const { return std::hypot(gMomHyp[0], gMomHyp[1]); }
  float genPtDau() const { return std::hypot(gMomDau[0], gMomDau[1]); }
  float genPhi() const { return std::atan2(gMomHyp[1], gMomHyp[0]); }
  float genEta() const { return std::asinh(gMomHyp[2] / genPt()); }

  int hyperTrackID;
  int dauTrackID;
  bool isMatter = false;

  std::array<float, 3> momHyp = {-999, -999, -999};
  std::array<float, 3> momDau = {-999, -999, -999};
  std::array<float, 3> primVtx = {-999, -999, -999};
  std::array<float, 3> decVtx = {-999, -999, -999};

  float dcaKinkTopo = -999;
  float nSigmaTPCDau = -999;
  float nSigmaTOFDau = -999;
  float dauDCAXY = -999;
  float hypDCAXY = -999;
  float kinkAngle = -999;

  float momDauTPC = -999.f;
  uint16_t tpcSignalDau = 0u;
  uint8_t nTPCClustersDau = 0u;
  uint8_t trackingPIDDaughter = 0u; // flags for triton PID in tracking

  uint32_t clusterSizeITSHyp = 0u;
  uint32_t clusterSizeITSDau = 0u;

  std::array<float, 3> gMomHyp;
  std::array<float, 3> gMomDau;
  std::array<float, 3> gDecVtx;

  bool isSignal = false;          // true MC signal
  bool isReco = false;            // true if the candidate is actually reconstructed
  float itsPt = -999.f;           // pt of the ITS hypertrack even when the topology is not reconstructed, tagged with the MC truth
  uint16_t mcMask = false;        // to study fake its tracks
  bool isRecoMCCollision = false; // true if the corresponding MC collision has been reconstructed
  bool isSurvEvSelection = false; // true if the corresponding event passed the event selection
  int pdgCode = 0;                // pdg code of the mother particle
};

struct hyperKinkRecoTask {

  Produces<aod::DataHypKinkCands> outputDataTable;
  Produces<aod::MCHypKinkCands> outputMCTable;
  Service<o2::ccdb::BasicCCDBManager> ccdb;

  // Selection criteria
  Configurable<float> invMassLow{"invMassLow", 3.7, "Lower limit for the invariant mass"};
  Configurable<float> invMassHigh{"invMassHigh", 15., "Upper limit for the invariant mass"};
  Configurable<float> maxDCAHypToPV{"maxDCAHypToPV", 0.1, "Max DCA of the mother track to the PV"};
  Configurable<float> minDCADauToPV{"minDCADauToPV", 0., "Min DCA of the charged daughter to the PV"};
  Configurable<float> minPtHyp{"minPtHyp", 0.5, "Minimum pT of the hypercandidate"};
  Configurable<float> maxZDiff{"maxZDiff", 20., "Max z difference between the kink daughter and the mother track"};
  Configurable<float> maxPhiDiff{"maxPhiDiff", 100, "Max phi difference between the kink daughter and the mother track"};
  Configurable<float> timeMarginNS{"timeMarginNS", 600, "Additional time res tolerance in ns"};
  Configurable<float> etaMax{"eta", 1., "eta daughter"};
  Configurable<float> nSigmaTPCCutDau{"nSigmaTPCCutDau", 5, "daughter dEdx cut (n sigma)"};
  Configurable<float> nSigmaTOFCutDau{"nSigmaTOFCutDau", 5, "daughter TOF cut (n sigma)"};
  Configurable<float> nTPCClusMinDau{"nTPCClusMinDau", 80, "daughter NTPC clusters cut"};
  Configurable<bool> alwaysAskTOF{"alwaysAskTOF", false, "If true, ask for TOF signal"};
  Configurable<bool> mcSignalOnly{"mcSignalOnly", true, "If true, save only signal in MC"};

  // Define o2 fitter, 2-prong, active memory (no need to redefine per event)
  o2::vertexing::DCAFitterN<2> fitter;
  o2::base::MatLayerCylSet* lut = nullptr;

  // constants
  float tritonMass = o2::constants::physics::MassTriton;
  float pi0Mass = o2::constants::physics::MassPi0;
  float alphaMass = o2::constants::physics::MassAlpha;
  float radToDeg = 180. / M_PI;
  Configurable<int> hyperPdg{"hyperPdg", 1110020040, "PDG code of the hyper-mother"};
  Configurable<int> dauPdg{"dauPdg", 1000020040, "PDG code of the kink daughter"};

  svPoolCreator svCreator{hyperPdg, dauPdg};

  // bethe bloch parameters
  Configurable<LabeledArray<double>> cfgBetheBlochParams{"cfgBetheBlochParams", {betheBlochDefault[0], 1, 6, particleNames, betheBlochParNames}, "TPC Bethe-Bloch parameterisation for Triton"};
  Configurable<int> cfgMaterialCorrection{"cfgMaterialCorrection", static_cast<int>(o2::base::Propagator::MatCorrType::USEMatCorrNONE), "Type of material correction"};
  Configurable<float> customVertexerTimeMargin{"customVertexerTimeMargin", 800, "Time margin for custom vertexer (ns)"};
  Configurable<bool> skipAmbiTracks{"skipAmbiTracks", false, "Skip ambiguous tracks"};
  Configurable<bool> unlikeSignBkg{"unlikeSignBkg", false, "Use unlike sign background"};

  // CCDB options
  Configurable<double> d_bz_input{"d_bz", -999, "bz field, -999 is automatic"};
  Configurable<std::string> ccdburl{"ccdb-url", "http://alice-ccdb.cern.ch", "url of the ccdb repository"};
  Configurable<std::string> grpPath{"grpPath", "GLO/GRP/GRP", "Path of the grp file"};
  Configurable<std::string> grpmagPath{"grpmagPath", "GLO/Config/GRPMagField", "CCDB path of the GRPMagField object"};
  Configurable<std::string> lutPath{"lutPath", "GLO/Param/MatLUT", "Path of the Lut parametrization"};
  Configurable<std::string> geoPath{"geoPath", "GLO/Config/GeometryAligned", "Path of the geometry file"};
  Configurable<std::string> pidPath{"pidPath", "", "Path to the PID response object"};

  // PDG codes

  // histogram axes
  ConfigurableAxis rigidityBins{"rigidityBins", {200, -10.f, 10.f}, "Binning for rigidity #it{p}^{TPC}/#it{z}"};
  ConfigurableAxis dedxBins{"dedxBins", {1000, 0.f, 1000.f}, "Binning for dE/dx"};
  ConfigurableAxis nSigmaBins{"nSigmaBins", {200, -5.f, 5.f}, "Binning for n sigma"};
  ConfigurableAxis zVtxBins{"zVtxBins", {100, -20.f, 20.f}, "Binning for n sigma"};
  ConfigurableAxis centBins{"centBins", {100, 0.f, 100.f}, "Binning for centrality"};

  std::vector<int> recoCollisionIds;
  std::vector<bool> isSurvEvSelCollision;
  std::vector<bool> goodCollision;
  // std vector of candidates
  std::vector<kinkCandidate> kinkCandidates;
  // vector to keep track of MC mothers already filled
  std::vector<unsigned int> filledMothers;
  // vector to keep track of the collisions passing the event selection in the MC
  HistogramRegistry qaRegistry{"QA", {}, OutputObjHandlingPolicy::AnalysisObject};

  int mRunNumber;
  float d_bz;
  std::array<float, 6> mBBparamsTrit;

  void init(InitContext const&)
  {

    mRunNumber = 0;
    d_bz = 0;

    ccdb->setURL(ccdburl);
    ccdb->setCaching(true);
    ccdb->setLocalObjectValidityChecking();
    ccdb->setFatalWhenNull(false);
    fitter.setPropagateToPCA(true);
    fitter.setMaxR(200.);
    fitter.setMinParamChange(1e-3);
    fitter.setMinRelChi2Change(0.9);
    fitter.setMaxDZIni(1e9);
    fitter.setMaxChi2(1e9);
    fitter.setUseAbsDCA(true);
    lut = o2::base::MatLayerCylSet::rectifyPtrFromFile(ccdb->get<o2::base::MatLayerCylSet>(lutPath));
    int mat{static_cast<int>(cfgMaterialCorrection)};
    fitter.setMatCorrType(static_cast<o2::base::Propagator::MatCorrType>(mat));

    svCreator.setTimeMargin(customVertexerTimeMargin);
    if (skipAmbiTracks) {
      svCreator.setSkipAmbiTracks();
    }

    const AxisSpec rigidityAxis{rigidityBins, "#it{p}^{TPC}/#it{z} (GeV/#it{c})"};
    const AxisSpec dedxAxis{dedxBins, "d#it{E}/d#it{x}"};
    const AxisSpec nSigma3HeAxis{nSigmaBins, "n_{#sigma}({}^{3}H)"};
    const AxisSpec zVtxAxis{zVtxBins, "z_{vtx} (cm)"};
    const AxisSpec centAxis{centBins, "Centrality"};

    hNsigmaDauSel = qaRegistry.add<TH2>("hNsigmaDauSel", "; p_{TPC}/z (GeV/#it{c}); n_{#sigma} ({}^{3}H)", HistType::kTH2F, {rigidityAxis, nSigma3HeAxis});
    hDeDxDauSel = qaRegistry.add<TH2>("hDeDxDauSel", ";p_{TPC}/z (GeV/#it{c}); dE/dx", HistType::kTH2F, {rigidityAxis, dedxAxis});
    hDeDxTot = qaRegistry.add<TH2>("hDeDxTot", ";p_{TPC}/z (GeV/#it{c}); dE/dx", HistType::kTH2F, {rigidityAxis, dedxAxis});
    hEvents = qaRegistry.add<TH1>("hEvents", ";Events; ", HistType::kTH1D, {{3, -0.5, 2.5}});
    hEvents->GetXaxis()->SetBinLabel(1, "All");
    hEvents->GetXaxis()->SetBinLabel(2, "sel8");
    hEvents->GetXaxis()->SetBinLabel(3, "z vtx");

    // if (doprocessMC) {
    //   hIsMatterGen = qaRegistry.add<TH1>("hIsMatterGen", ";; ", HistType::kTH1D, {{2, -0.5, 1.5}});
    //   hIsMatterGen->GetXaxis()->SetBinLabel(1, "Matter");
    //   hIsMatterGen->GetXaxis()->SetBinLabel(2, "Antimatter");
    // }
    hZvtx = qaRegistry.add<TH1>("hZvtx", ";z_{vtx} (cm); ", HistType::kTH1D, {{100, -20, 20}});
    hCentFT0M = qaRegistry.add<TH1>("hCentFT0M", ";Centrality; ", HistType::kTH1D, {{100, 0, 100}});
    hInvMass = qaRegistry.add<TH1>("hInvMass", ";Invariant mass (GeV/#it{c}^{2}); ", HistType::kTH1D, {{32, invMassLow, invMassLow + 0.8}});
    hHyperPt = qaRegistry.add<TH1>("hHyperPt", ";p_{T} (GeV/#it{c}); ", HistType::kTH1D, {{100, 0, 10}});
    hCandidates = qaRegistry.add<TH1>("hCandidates", ";Candidates; ", HistType::kTH1D, {{10, -0.5, 9.5}});
    hTrueCandidates = qaRegistry.add<TH1>("hTrueCandidates", ";Candidates; ", HistType::kTH1D, {{10, -0.5, 9.5}});
    hTracks = qaRegistry.add<TH1>("hTracks", ";Tracks; ", HistType::kTH1D, {{10, -0.5, 9.5}});
    hTracks->GetXaxis()->SetBinLabel(1, "All");
    hTracks->GetXaxis()->SetBinLabel(2, "isHyper");
    hTracks->GetXaxis()->SetBinLabel(3, "isMcHyper");
    hTracks->GetXaxis()->SetBinLabel(4, "isHyper && isMCHyper");
    hTracks->GetXaxis()->SetBinLabel(5, "isDau");
    hTracks->GetXaxis()->SetBinLabel(6, "isMcDau");
    hTracks->GetXaxis()->SetBinLabel(7, "isMcDauSignal");
    hTracks->GetXaxis()->SetBinLabel(8, "isDau && isMcDau");

    hDiffHyperPt = qaRegistry.add<TH1>("hDiffHyperPt", ";#Delta p_{T} (GeV/#it{c}); ", HistType::kTH1D, {{200, -10, 10}});
    hDiffHyperP = qaRegistry.add<TH1>("hDiffHyperP", ";#Delta p (GeV/#it{c}); ", HistType::kTH1D, {{200, -10, 10}});
    hRVsDiffHyperPt = qaRegistry.add<TH2>("hRVsDiffHyperPt", "; #Delta p_{T} (GeV/#it{c}); R (cm);", HistType::kTH2F, {{200, -10, 10}, {40, 0, 40}});
    hRVsDiffTrueHyperPt = qaRegistry.add<TH2>("hRVsDiffTrueHyperPt", "; #Delta p_{T} (GeV/#it{c}); R (cm);", HistType::kTH2F, {{200, -10, 10}, {40, 0, 40}});
    hDiffDauPx = qaRegistry.add<TH1>("hDiffDauPx", ";#Delta p_{x} (GeV/#it{c}); ", HistType::kTH1D, {{200, -10, 10}});
    hDiffDauPy = qaRegistry.add<TH1>("hDiffDauPy", ";#Delta p_{y} (GeV/#it{c}); ", HistType::kTH1D, {{200, -10, 10}});
    hDiffDauPz = qaRegistry.add<TH1>("hDiffDauPz", ";#Delta p_{z} (GeV/#it{c}); ", HistType::kTH1D, {{200, -10, 10}});
    hDiffSVx = qaRegistry.add<TH1>("hDiffSVx", ";#Delta x (cm); ", HistType::kTH1D, {{200, -10, 10}});
    hDiffSVy = qaRegistry.add<TH1>("hDiffSVy", ";#Delta y (cm); ", HistType::kTH1D, {{200, -10, 10}});
    hDiffSVz = qaRegistry.add<TH1>("hDiffSVz", ";#Delta z (cm); ", HistType::kTH1D, {{200, -10, 10}});
    hHyperITSNcls = qaRegistry.add<TH1>("hHyperITSNcls", ";ITS Ncls; ", HistType::kTH1D, {{8, -0.5, 7.5}});
    hTrueHyperITSNcls = qaRegistry.add<TH1>("hTrueHyperITSNcls", ";ITS Ncls; ", HistType::kTH1D, {{8, -0.5, 7.5}});
  }

  template <class Tcoll>
  void selectGoodCollisions(const Tcoll& collisions)
  {
    for (const auto& collision : collisions) {
      auto bc = collision.template bc_as<aod::BCsWithTimestamps>();
      initCCDB(bc);
      hEvents->Fill(0.);

      if (!collision.sel8() || std::abs(collision.posZ()) > 10) {
        continue;
      }

      goodCollision[collision.globalIndex()] = true;
      hEvents->Fill(1.);
      hZvtx->Fill(collision.posZ());
      hCentFT0M->Fill(collision.centFT0M());
    }
  }

  template <class Tcoll>
  void selectGoodCollisionsMC(const Tcoll& collisions)
  {
    for (const auto& collision : collisions) {
      auto bc = collision.template bc_as<aod::BCsWithTimestamps>();
      initCCDB(bc);
      hEvents->Fill(0.);
      if (collision.has_mcCollision()) {
        recoCollisionIds[collision.mcCollisionId()] = collision.globalIndex();
      }
      /*if (!collision.selection_bit(aod::evsel::kIsTriggerTVX) || !collision.selection_bit(aod::evsel::kNoTimeFrameBorder) || std::abs(collision.posZ()) > 10)
        continue;*/
      /*if (std::abs(collision.posZ()) > 10)
        continue;*/

      if (collision.has_mcCollision()) {
        isSurvEvSelCollision[collision.mcCollisionId()] = true;
      }
      goodCollision[collision.globalIndex()] = true;

      hEvents->Fill(1.);
      hZvtx->Fill(collision.posZ());
      /*LOG(info) << "Fill hCentFT0M";
      hCentFT0A->Fill(collision.centFT0A());
      hCentFT0C->Fill(collision.centFT0C());
      hCentFT0M->Fill(collision.centFT0M());*/
    }
  }

  float angleCutFunction(float x)
  {
    float par1 = 2.99131; // hypertriton mass
    float par2 = 0.07;    // optimized by mdiotti
    float par3 = TMath::Pi();

    if (hyperPdg == 1110020040) {
      par1 = o2::constants::physics::MassHyperHelium4Sigma; // hyperhe4s mass
    }
    return par1 * (par2 / (sqrt((x * x) * (1 - (par2 * par2)) - ((par1 * par1) * (par2 * par2))))) * (180. / par3) + 1;
  }

  template <typename T>
  float computeNSigmaTrit(const T& candidate)
  {
    float nSigmaTrit = -100.f;
    if (mBBparamsTrit[5] > 0) {
      float expTPCSignal = o2::tpc::BetheBlochAleph(static_cast<float>(candidate.tpcInnerParam() * 2 / constants::physics::MassHelium3), mBBparamsTrit[0], mBBparamsTrit[1], mBBparamsTrit[2], mBBparamsTrit[3], mBBparamsTrit[4]);
      double resoTPC{expTPCSignal * mBBparamsTrit[5]};
      nSigmaTrit = static_cast<float>((candidate.tpcSignal() - expTPCSignal) / resoTPC);
    } else {
      nSigmaTrit = candidate.tpcNSigmaTr();
    }
    return nSigmaTrit;
  }

  template <typename T>
  bool selectHyperTrack(const T& candidate)
  {
    if (candidate.hasITS() && !candidate.hasTPC() && !candidate.hasTOF() && candidate.itsNCls() < 6 &&
        candidate.itsNClsInnerBarrel() == 3 && candidate.itsChi2NCl() < 36 && candidate.pt() > minPtHyp) {
      hHyperPt->Fill(candidate.pt());
      hHyperITSNcls->Fill(candidate.itsNCls());
      return true;
    }

    return false;
  }

  template <typename T>
  bool selectDauTrack(const T& candidate)
  {
    if (!candidate.hasTPC() || !candidate.hasITS()) {
      return false;
    }

    if (alwaysAskTOF && !candidate.hasTOF()) {
      return false;
    }

    float nSigmaDau = -100.f, nSigmaTOFDau = -100.f;
    if (dauPdg == 1000010030) {
      nSigmaDau = computeNSigmaTrit(candidate);
      nSigmaTOFDau = candidate.tofNSigmaTr();
    } else if (dauPdg == 1000020040) {
      nSigmaDau = candidate.tpcNSigmaAl();
      nSigmaTOFDau = candidate.tofNSigmaAl();
    }

    bool isGoodTPCCand = false;
    if (candidate.itsNClsInnerBarrel() == 0 && candidate.itsNCls() <= 4 &&
        candidate.tpcNClsCrossedRows() >= 70 && candidate.tpcChi2NCl() < 4.f &&
        candidate.tpcNClsCrossedRows() > 0.8 * candidate.tpcNClsFindable() && candidate.tpcNClsFound() > 80 && std::abs(nSigmaDau) < nSigmaTPCCutDau) {
      isGoodTPCCand = true;
    }

    if (!isGoodTPCCand) {
      return false;
    }

    if (candidate.hasTOF() && std::abs(nSigmaTOFDau) > nSigmaTOFCutDau) {
      return false;
    }

    hNsigmaDauSel->Fill(candidate.pt(), nSigmaDau);
    hDeDxDauSel->Fill(candidate.tpcInnerParam(), candidate.tpcSignal());

    return true;
  }

  template <class Tcolls, class Ttracks>
  // void fillCandidateData(const Tcolls& collisions, const Ttracks& tracks, aod::AmbiguousTracks const& ambiguousTracks, aod::BCsWithTimestamps const& bcs)
  void fillCandidateData(const Tcolls& collisions, const Ttracks& tracks, aod::McTrackLabels const& trackLabelsMC, aod::AmbiguousTracks const& ambiguousTracks, aod::BCsWithTimestamps const& bcs, aod::McParticles const&)
  {
    svCreator.clearPools();
    svCreator.fillBC2Coll(collisions, bcs);
    for (auto& track : tracks) {
      if (std::abs(track.eta()) > etaMax)
        continue;

      bool isDau = selectDauTrack(track);
      bool isHyp = selectHyperTrack(track);

      hTracks->Fill(0.);
      if (isHyp) {
        hTracks->Fill(1.);
      }

      if (isDau) {
        hTracks->Fill(4.);
      }

      bool isMCHyper = false, isMCDau = false;
      auto mcLabel = trackLabelsMC.rawIteratorAt(track.globalIndex());
      if (mcLabel.has_mcParticle()) {
        auto mcPart = mcLabel.template mcParticle_as<aod::McParticles>();
        if (std::abs(mcPart.pdgCode()) == hyperPdg) {
          hTracks->Fill(2.);
          hTrueHyperITSNcls->Fill(track.itsNCls());
          float svR = -999;
          for (auto& dauMCTracks : mcPart.template daughters_as<aod::McParticles>()) {
            if (std::abs(dauMCTracks.pdgCode()) == dauPdg) {
              svR = std::hypot(dauMCTracks.vx(), dauMCTracks.vy());
              hRVsDiffTrueHyperPt->Fill(mcPart.pt() - 2 * track.pt(), svR);
            }
          }
          LOG(info) << "Hyper candidate found";
          isMCHyper = true;
        }
        if (std::abs(mcPart.pdgCode()) == dauPdg) {
          hTracks->Fill(5.);
          for (auto& dauMcMother : mcPart.template mothers_as<aod::McParticles>()) {
            if (std::abs(dauMcMother.pdgCode()) == hyperPdg) {
              hTracks->Fill(6.);
              LOG(info) << "Daughter candidate found";
              isMCDau = true;
            }
          }
        }
      }

      if (isHyp && isMCHyper) {
        hTracks->Fill(3.);
      }

      if (isDau && isMCDau) {
        hTracks->Fill(7.);
      }

      if (!isDau && !isHyp)
        continue;

      if (!isMCHyper && !isMCDau) {
        continue;
      }

      if (isHyp && !track.has_collision()) {
        LOG(info) << "Hyper candidate without collision";
        continue;
      }

      hTracks->Fill(8.);

      int pdgHypo = isHyp ? hyperPdg : dauPdg;
      LOG(info) << "Start appending track";
      svCreator.appendTrackCand(track, collisions, pdgHypo, ambiguousTracks, bcs);
    }
    auto& kinkPool = svCreator.getSVCandPool(collisions, !unlikeSignBkg);
    LOG(info) << "SV pool size: " << kinkPool.size();

    for (auto& svCand : kinkPool) {

      kinkCandidate kinkCand;
      hCandidates->Fill(0.);

      auto trackHyper = tracks.rawIteratorAt(svCand.tr0Idx);
      auto trackDaughter = tracks.rawIteratorAt(svCand.tr1Idx);

      auto mcLabelHyper = trackLabelsMC.rawIteratorAt(trackHyper.globalIndex());
      auto mcLabelDau = trackLabelsMC.rawIteratorAt(trackDaughter.globalIndex());

      bool isSignal = false;
      float mcHyperP[3], mcDauP[3], mcSvPos[3], mcHyperPt, mcHyperPTotal;
      if (mcLabelHyper.has_mcParticle() && mcLabelDau.has_mcParticle()) {
        auto mcPartHyper = mcLabelHyper.template mcParticle_as<aod::McParticles>();
        auto mcPartDau = mcLabelDau.template mcParticle_as<aod::McParticles>();
        if (std::abs(mcPartHyper.pdgCode()) == hyperPdg && std::abs(mcPartDau.pdgCode()) == dauPdg) {
          for (auto& dauMCTracks : mcPartHyper.template daughters_as<aod::McParticles>()) {
            if (std::abs(dauMCTracks.pdgCode()) == dauPdg) {
              if (dauMCTracks.globalIndex() == mcPartDau.globalIndex()) {
                isSignal = true;
                mcSvPos[0] = mcPartDau.vx();
                mcSvPos[1] = mcPartDau.vy();
                mcSvPos[2] = mcPartDau.vz();
                mcHyperP[0] = mcPartHyper.px();
                mcHyperP[1] = mcPartHyper.py();
                mcHyperP[2] = mcPartHyper.pz();
                mcHyperPt = mcPartHyper.pt();
                mcHyperPTotal = mcPartHyper.p();
                mcDauP[0] = mcPartDau.px();
                mcDauP[1] = mcPartDau.py();
                mcDauP[2] = mcPartDau.pz();
              }
            }
          }
        }
      }

      if (isSignal) {
        hTrueCandidates->Fill(0.);
      } else {
        continue;
      }

      auto const& collision = trackHyper.template collision_as<Tcolls>();
      o2::dataformats::VertexBase primaryVertex;
      primaryVertex.setPos({collision.posX(), collision.posY(), collision.posZ()});
      primaryVertex.setCov(collision.covXX(), collision.covXY(), collision.covYY(), collision.covXZ(), collision.covYZ(), collision.covZZ());
      kinkCand.primVtx = {primaryVertex.getX(), primaryVertex.getY(), primaryVertex.getZ()};

      o2::track::TrackParCov trackParCovHyper = getTrackParCov(trackHyper);
      o2::base::Propagator::Instance()->PropagateToXBxByBz(trackParCovHyper, LayerRadii[trackHyper.itsNCls() - 1]);

      o2::track::TrackParCov trackParCovHyperPV = getTrackParCov(trackHyper);
      gpu::gpustd::array<float, 2> dcaInfoHyp;
      o2::base::Propagator::Instance()->propagateToDCABxByBz({primaryVertex.getX(), primaryVertex.getY(), primaryVertex.getZ()}, trackParCovHyperPV, 2.f, static_cast<o2::base::Propagator::MatCorrType>(cfgMaterialCorrection.value), &dcaInfoHyp);

      if (std::abs(dcaInfoHyp[0]) > maxDCAHypToPV) {
        continue;
      }
      hCandidates->Fill(1.);
      if (isSignal) {
        hTrueCandidates->Fill(1.);
      }

      o2::track::TrackParCov trackParCovDau = getTrackParCov(trackDaughter);

      // check if the kink daughter is close to the hypertriton
      if (std::abs(trackParCovHyper.getZ() - trackParCovDau.getZ()) > maxZDiff) {
        continue;
      }
      hCandidates->Fill(2.);
      if (isSignal) {
        hTrueCandidates->Fill(2.);
      }

      if ((std::abs(trackParCovHyper.getPhi() - trackParCovDau.getPhi()) * radToDeg) > maxPhiDiff) {
        continue;
      }
      hCandidates->Fill(3.);
      if (isSignal) {
        hTrueCandidates->Fill(3.);
      }

      // propagate to PV
      gpu::gpustd::array<float, 2> dcaInfoDau;
      o2::base::Propagator::Instance()->propagateToDCABxByBz({primaryVertex.getX(), primaryVertex.getY(), primaryVertex.getZ()}, trackParCovDau, 2.f, static_cast<o2::base::Propagator::MatCorrType>(cfgMaterialCorrection.value), &dcaInfoDau);
      if (std::abs(dcaInfoDau[0]) < minDCADauToPV) {
        continue;
      }
      hCandidates->Fill(4.);
      if (isSignal) {
        hTrueCandidates->Fill(4.);
      }

      int nCand = 0;
      try {
        nCand = fitter.process(trackParCovHyper, trackParCovDau);
      } catch (...) {
        LOG(error) << "Exception caught in DCA fitter process call!";
        continue;
      }
      if (nCand == 0) {
        continue;
      }
      hCandidates->Fill(5.);
      if (isSignal) {
        hTrueCandidates->Fill(5.);
      }

      if (!fitter.propagateTracksToVertex()) {
        continue;
      }
      hCandidates->Fill(6.);
      if (isSignal) {
        hTrueCandidates->Fill(6.);
      }

      auto propHyperTrack = fitter.getTrack(0);
      auto propDauTrack = fitter.getTrack(1);

      kinkCand.decVtx = fitter.getPCACandidatePos();

      hDiffSVx->Fill(kinkCand.decVtx[0] - mcSvPos[0]);
      hDiffSVy->Fill(kinkCand.decVtx[1] - mcSvPos[1]);
      hDiffSVz->Fill(kinkCand.decVtx[2] - mcSvPos[2]);

      // cut on decay radius to 17 cm
      if (kinkCand.decVtx[0] * kinkCand.decVtx[0] + kinkCand.decVtx[1] * kinkCand.decVtx[1] < 17 * 17) {
        continue;
      }
      hCandidates->Fill(7.);
      if (isSignal) {
        hTrueCandidates->Fill(7.);
      }

      propHyperTrack.getPxPyPzGlo(kinkCand.momHyp);
      propDauTrack.getPxPyPzGlo(kinkCand.momDau);
      if (hyperPdg == 1110020040) {
        for (int i = 0; i < 3; i++) {
          kinkCand.momHyp[i] *= 2;
        }
      }
      if (dauPdg == 1000020040) {
        for (int i = 0; i < 3; i++) {
          kinkCand.momDau[i] *= 2;
        }
      }
      hDiffHyperPt->Fill(mcHyperPt - 2 * propHyperTrack.getPt());
      hDiffHyperP->Fill(mcHyperPTotal - 2 * propHyperTrack.getP());
      hRVsDiffHyperPt->Fill(mcHyperPt - 2 * propHyperTrack.getPt(), std::sqrt(mcSvPos[0] * mcSvPos[0] + mcSvPos[1] * mcSvPos[1]));
      hDiffDauPx->Fill(mcDauP[0] - kinkCand.momDau[0]);
      hDiffDauPy->Fill(mcDauP[1] - kinkCand.momDau[1]);
      hDiffDauPz->Fill(mcDauP[2] - kinkCand.momDau[2]);

      float pHyp = propHyperTrack.getP();
      float pDau = propDauTrack.getP();
      if (hyperPdg == 1110020040) {
        pHyp *= 2;
      }
      if (dauPdg == 1000020040) {
        pDau *= 2;
      }
      float spKink = kinkCand.momHyp[0] * kinkCand.momDau[0] + kinkCand.momHyp[1] * kinkCand.momDau[1] + kinkCand.momHyp[2] * kinkCand.momDau[2];
      kinkCand.kinkAngle = std::acos(spKink / (pHyp * pDau));
      float angleCut = angleCutFunction(pHyp);
      if (kinkCand.kinkAngle * radToDeg > angleCut) {
        continue;
      }
      hCandidates->Fill(8.);
      if (isSignal) {
        hTrueCandidates->Fill(8.);
      }

      std::array<float, 3> pi0mom{0.f, 0.f, 0.f};
      for (int i = 0; i < 3; i++) {
        pi0mom[i] = kinkCand.momHyp[i] - kinkCand.momDau[i];
      }
      float pi0E = std::sqrt(pi0mom[0] * pi0mom[0] + pi0mom[1] * pi0mom[1] + pi0mom[2] * pi0mom[2] + pi0Mass * pi0Mass);

      float dauE = -999;
      if (dauPdg == 1000010030) {
        dauE = std::sqrt(pDau * pDau + tritonMass * tritonMass);
      } else if (dauPdg == 1000020040) {
        dauE = std::sqrt(pDau * pDau + alphaMass * alphaMass);
      }
      float invMass = std::sqrt((pi0E + dauE) * (pi0E + dauE) - pHyp * pHyp);

      if (invMass < invMassLow || invMass > invMassHigh) {
        continue;
      }
      hCandidates->Fill(9.);
      if (isSignal) {
        hTrueCandidates->Fill(9.);
      }

      if (dauPdg == 1000010030) {
        kinkCand.nSigmaTPCDau = computeNSigmaTrit(trackDaughter);
      } else if (dauPdg == 1000020040) {
        kinkCand.nSigmaTPCDau = trackDaughter.tpcNSigmaAl();
      }

      kinkCand.hyperTrackID = trackHyper.globalIndex();
      kinkCand.hypDCAXY = dcaInfoHyp[0];
      kinkCand.clusterSizeITSHyp = trackHyper.itsClusterSizes();
      kinkCand.isMatter = trackHyper.sign() > 0;
      kinkCand.dauTrackID = trackDaughter.globalIndex();
      kinkCand.dauDCAXY = dcaInfoDau[0];
      kinkCand.clusterSizeITSDau = trackDaughter.itsClusterSizes();
      kinkCand.nTPCClustersDau = trackDaughter.tpcNClsFound();
      kinkCand.tpcSignalDau = trackDaughter.tpcSignal();
      kinkCand.momDauTPC = trackDaughter.tpcInnerParam();
      kinkCand.dcaKinkTopo = std::sqrt(fitter.getChi2AtPCACandidate());
      kinkCand.trackingPIDDaughter = trackDaughter.pidForTracking();
      kinkCand.isReco = true;
      kinkCandidates.push_back(kinkCand);

      hInvMass->Fill(invMass);
    }
  }

  void fillMCinfo(aod::McTrackLabels const& trackLabels, aod::McParticles const&)
  {

    for (auto& kinkCand : kinkCandidates) {
      auto mcLabHyper = trackLabels.rawIteratorAt(kinkCand.hyperTrackID);
      auto mcLabDau = trackLabels.rawIteratorAt(kinkCand.dauTrackID);
      if (mcLabHyper.has_mcParticle() && mcLabDau.has_mcParticle()) {
        auto mcTrackHyper = mcLabHyper.mcParticle_as<aod::McParticles>();
        auto mcTrackDaughter = mcLabDau.mcParticle_as<aod::McParticles>();

        if (std::abs(mcTrackHyper.pdgCode()) != hyperPdg || std::abs(mcTrackDaughter.pdgCode()) != dauPdg) {
          continue;
        }
        auto dauIdx = mcTrackDaughter.globalIndex();
        kinkCand.isSignal = false;
        for (auto& dauMCTracks : mcTrackHyper.daughters_as<aod::McParticles>()) {
          if (std::abs(dauMCTracks.pdgCode()) == dauPdg) {
            if (dauMCTracks.globalIndex() == dauIdx) {
              kinkCand.isSignal = true;
              break;
            }
          }
        }
        auto primVtx = array{mcTrackHyper.vx(), mcTrackHyper.vy(), mcTrackHyper.vz()};
        auto secVtx = array{mcTrackDaughter.vx(), mcTrackDaughter.vy(), mcTrackDaughter.vz()};
        for (int i = 0; i < 3; i++) {
          kinkCand.gDecVtx[i] = secVtx[i] - primVtx[i];
        }
        kinkCand.pdgCode = mcTrackHyper.pdgCode();
        kinkCand.mcMask = mcLabHyper.mcMask();
        filledMothers.push_back(mcTrackHyper.globalIndex());
      }
    }
  }

  void initCCDB(aod::BCsWithTimestamps::iterator const& bc)
  {
    if (mRunNumber == bc.runNumber()) {
      return;
    }
    auto run3grp_timestamp = bc.timestamp();

    o2::parameters::GRPObject* grpo = ccdb->getForTimeStamp<o2::parameters::GRPObject>(grpPath, run3grp_timestamp);
    o2::parameters::GRPMagField* grpmag = 0x0;
    if (grpo) {
      o2::base::Propagator::initFieldFromGRP(grpo);
      if (d_bz_input < -990) {
        // Fetch magnetic field from ccdb for current collision
        d_bz = grpo->getNominalL3Field();
        LOG(info) << "Retrieved GRP for timestamp " << run3grp_timestamp << " with magnetic field of " << d_bz << " kZG";
      } else {
        d_bz = d_bz_input;
      }
    } else {
      grpmag = ccdb->getForTimeStamp<o2::parameters::GRPMagField>(grpmagPath, run3grp_timestamp);
      if (!grpmag) {
        LOG(fatal) << "Got nullptr from CCDB for path " << grpmagPath << " of object GRPMagField and " << grpPath << " of object GRPObject for timestamp " << run3grp_timestamp;
      }
      o2::base::Propagator::initFieldFromGRP(grpmag);
      if (d_bz_input < -990) {
        // Fetch magnetic field from ccdb for current collision
        d_bz = std::lround(5.f * grpmag->getL3Current() / 30000.f);
        LOG(info) << "Retrieved GRP for timestamp " << run3grp_timestamp << " with magnetic field of " << d_bz << " kZG";
      } else {
        d_bz = d_bz_input;
      }
    }
    if (!pidPath.value.empty()) {
      auto tritpid = ccdb->getForTimeStamp<std::array<float, 6>>(pidPath.value + "_Trit", run3grp_timestamp);
      std::copy(tritpid->begin(), tritpid->end(), mBBparamsTrit.begin());
    } else {
      for (int i = 0; i < 5; i++) {
        mBBparamsTrit[i] = cfgBetheBlochParams->get("Triton", Form("p%i", i));
      }
      mBBparamsTrit[5] = cfgBetheBlochParams->get("Triton", "resolution");
    }
    fitter.setBz(d_bz);
    mRunNumber = bc.runNumber();

    o2::base::Propagator::Instance()->setMatLUT(lut);
  }

  void processData(CollisionsFull const& collisions, TracksFull const& tracks, aod::AmbiguousTracks const& ambiTracks, aod::BCsWithTimestamps const& bcs)
  {
    if (mBBparamsTrit[5] < 0) {
      LOG(info) << "Bethe-Bloch parameters for Triton not set, using default nSigma";
    }
    goodCollision.clear();
    goodCollision.resize(collisions.size(), false);
    kinkCandidates.clear();
    selectGoodCollisions(collisions);
    /*fillCandidateData(collisions, tracks, ambiTracks, bcs);
    for (auto& kinkCand : kinkCandidates) {
      outputDataTable(kinkCand.primVtx[0], kinkCand.primVtx[1], kinkCand.primVtx[2],
                      kinkCand.decVtx[0], kinkCand.decVtx[1], kinkCand.decVtx[2],
                      kinkCand.isMatter, kinkCand.recoPtHyp(), kinkCand.recoPhiHyp(), kinkCand.recoEtaHyp(),
                      kinkCand.recoPtDau(), kinkCand.recoPhiDau(), kinkCand.recoEtaDau(),
                      kinkCand.hypDCAXY, kinkCand.dauDCAXY, kinkCand.dcaKinkTopo,
                      kinkCand.clusterSizeITSHyp, kinkCand.clusterSizeITSDau, kinkCand.trackingPIDDaughter,
                      kinkCand.momDauTPC, kinkCand.tpcSignalDau, kinkCand.nSigmaTPCDau, kinkCand.nSigmaTOFDau);
    }*/
  }
  PROCESS_SWITCH(hyperKinkRecoTask, processData, "Data analysis", true);

  // void processMC(CollisionsFullMC const& collisions, aod::McCollisions const& mcCollisions, TracksFull const& tracks, aod::AmbiguousTracks const& ambiTracks, aod::BCsWithTimestamps const& bcs, aod::McTrackLabels const& trackLabelsMC, aod::McParticles const& particlesMC)
  void processMC(CollisionsFullMC const& collisions, aod::McCollisions const& mcCollisions, TracksFull const& tracks, aod::AmbiguousTracks const& ambiTracks, aod::BCsWithTimestamps const& bcs, aod::McTrackLabels const& trackLabelsMC, aod::McParticles const& particlesMC)
  {
    filledMothers.clear();
    recoCollisionIds.clear();
    recoCollisionIds.resize(mcCollisions.size(), -1);
    isSurvEvSelCollision.clear();
    isSurvEvSelCollision.resize(mcCollisions.size(), false);
    goodCollision.clear();
    goodCollision.resize(collisions.size(), false);
    kinkCandidates.clear();

    selectGoodCollisionsMC(collisions);
    LOG(info) << "Good collisions: " << std::count(goodCollision.begin(), goodCollision.end(), true);
    // fillCandidateData(collisions, tracks, ambiTracks, bcs);
    fillCandidateData(collisions, tracks, trackLabelsMC, ambiTracks, bcs, particlesMC);
    LOG(info) << "Kink candidates: " << kinkCandidates.size();
    fillMCinfo(trackLabelsMC, particlesMC);

    std::vector<int> mcToKinkCandidates;
    mcToKinkCandidates.resize(particlesMC.size(), -1);

    for (auto& mcPart : particlesMC) {

      if (std::abs(mcPart.pdgCode()) != hyperPdg)
        continue;
      std::array<float, 3> secVtx;
      std::array<float, 3> primVtx = {mcPart.vx(), mcPart.vy(), mcPart.vz()};
      std::array<float, 3> momMother = {mcPart.px(), mcPart.py(), mcPart.pz()};
      std::array<float, 3> momDau;
      bool isDauFound = false;
      int hyperMCIndex = mcPart.globalIndex();
      for (auto& mcDaught : mcPart.daughters_as<aod::McParticles>()) {
        if (std::abs(mcDaught.pdgCode()) == dauPdg) {
          secVtx = {mcDaught.vx(), mcDaught.vy(), mcDaught.vz()};
          momDau = {mcDaught.px(), mcDaught.py(), mcDaught.pz()};
          isDauFound = true;
          break;
        }
      }
      if (!isDauFound) {
        continue;
      }

      if (std::find(filledMothers.begin(), filledMothers.end(), mcPart.globalIndex()) != std::end(filledMothers)) {
        continue;
      }
      kinkCandidate kinkCand;
      kinkCand.pdgCode = mcPart.pdgCode();
      kinkCand.isRecoMCCollision = recoCollisionIds[mcPart.mcCollisionId()] > 0;
      kinkCand.isSurvEvSelection = isSurvEvSelCollision[mcPart.mcCollisionId()];
      for (int i = 0; i < 3; i++) {
        kinkCand.gDecVtx[i] = secVtx[i] - primVtx[i];
        kinkCand.gMomHyp[i] = momMother[i];
        kinkCand.gMomDau[i] = momDau[i];
      }
      kinkCand.hyperTrackID = -1;
      kinkCand.dauTrackID = -1;
      kinkCand.isSignal = true;
      kinkCandidates.push_back(kinkCand);
      mcToKinkCandidates[hyperMCIndex] = kinkCandidates.size() - 1;
    }

    // look for hypernuclei or daughter tracks, findable part!
    for (auto& track : tracks) {
      auto mcLabel = trackLabelsMC.rawIteratorAt(track.globalIndex());
      if (mcLabel.has_mcParticle()) {
        auto mcTrack = mcLabel.mcParticle_as<aod::McParticles>();
        if (mcToKinkCandidates[mcTrack.globalIndex()] < 0 || !track.hasITS()) {
          continue;
        }
        auto& kinkCand = kinkCandidates[mcToKinkCandidates[mcTrack.globalIndex()]];
        kinkCand.mcMask = mcLabel.mcMask();
        kinkCand.itsPt = track.pt();
      }
    }

    for (auto& kinkCand : kinkCandidates) {
      if (!kinkCand.isSignal && mcSignalOnly) {
        continue;
      }
      int chargeFactor = -1 + 2 * (kinkCand.pdgCode > 0);
      outputMCTable(kinkCand.primVtx[0], kinkCand.primVtx[1], kinkCand.primVtx[2],
                    kinkCand.decVtx[0], kinkCand.decVtx[1], kinkCand.decVtx[2],
                    kinkCand.isMatter, kinkCand.recoPtHyp(), kinkCand.recoPhiHyp(), kinkCand.recoEtaHyp(),
                    kinkCand.recoPtDau(), kinkCand.recoPhiDau(), kinkCand.recoEtaDau(),
                    kinkCand.hypDCAXY, kinkCand.dauDCAXY, kinkCand.dcaKinkTopo,
                    kinkCand.clusterSizeITSHyp, kinkCand.clusterSizeITSDau, kinkCand.trackingPIDDaughter,
                    kinkCand.momDauTPC, kinkCand.tpcSignalDau, kinkCand.nSigmaTPCDau, kinkCand.nSigmaTOFDau,
                    kinkCand.gDecVtx[0], kinkCand.gDecVtx[1], kinkCand.gDecVtx[2],
                    kinkCand.genPt() * chargeFactor, kinkCand.genPtDau(),
                    kinkCand.isReco, kinkCand.isSignal, kinkCand.mcMask, kinkCand.itsPt,
                    kinkCand.isRecoMCCollision, kinkCand.isSurvEvSelection);
    }
  }
  PROCESS_SWITCH(hyperKinkRecoTask, processMC, "MC analysis", false);
};

WorkflowSpec
  defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<hyperKinkRecoTask>(cfgc)};
}
