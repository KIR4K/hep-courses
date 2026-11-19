#!/usr/bin/env python3

import os
import sys
import argparse

import ROOT
import pythia8


def get_system_name(cmnd_file_path: str) -> str:
    """
    Читает .cmnd файл и ищет строку '! System: <name>'.
    Возвращает <name> (например, 'pp5020', 'pPb5020', 'PbPb5020').
    Если не найдена — 'unknown'.
    """
    with open(cmnd_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("! System:"):
                return line.split(":", 1)[1].strip()
    return "unknown"


def run_pythia(input_file_name: str, n_events: int, seed: int) -> int:

    if not os.path.isfile(input_file_name):
        print(f"\033[1;31mError:\033[0m file {input_file_name} was not found")
        return 1

    pythia = pythia8.Pythia()
    pythia.readString("Print:quiet = on")

    # ---------- defaults (can be overridden by .cmnd) ----------
    # HardQCD needed for pT > ~3 GeV part of the spectrum
    pythia.readString("HardQCD:all = on")
    # MB baseline (elastic/diffractive not needed for central analysis)
    pythia.readString("SoftQCD:nonDiffractive = on")
    # CRITICAL: cut the 1/pT^4 divergence. Task requires pTHatMin = 3 GeV.
    pythia.readString("PhaseSpace:pTHatMin = 3.0")

    # seed from CLI (for parallel runs)
    pythia.readString("Random:setSeed = on")
    pythia.readString("Random:seed = " + str(seed))

    # ---------- read physics config from .cmnd ----------
    # Beams, energy, PDF, and anything else go here and OVERRIDE the defaults above
    pythia.readFile(input_file_name)

    pythia.init()

    system_name = get_system_name(input_file_name)
    print(f"\033[1;32m[INFO]\033[0m System = {system_name}, seed = {seed}")

    os.makedirs("output", exist_ok=True)
    output_file = ROOT.TFile(f"output/{system_name}_seed{seed}.root", "RECREATE")

    # N_ch forward multiplicity (ALICE-like window): 2.8 < |eta| < 5.1.
    # Range up to 5000 covers central Pb+Pb.
    # N_ch range depends on the system (pp ~ 5-15, pPb ~ 10-300, PbPb ~ 500-4000)
    if "PbPb" in system_name:
        nch_max, nch_bins = 10000.0, 2000   # bin width 5
    elif "pPb" in system_name:
        nch_max, nch_bins =  1500.0,  750   # bin width 2
    else:  # pp
        nch_max, nch_bins =   300.0,  300   # bin width 1

    hist_nch = ROOT.TH1D("N_ch", ";N_{ch};Events", nch_bins, 0., nch_max)
    hist_pt_vs_nch = ROOT.TH2D("pT_vs_Nch", ";p_{T} [GeV/c];N_{ch}",
                           250, 0., 25.,
                           nch_bins, 0., nch_max)

    # 1D diagnostic pT spectrum for protons (raw, sum of p and pbar)
    hist_pt_proton = ROOT.TH1D("pT_proton", ";p_{T} [GeV/c];Counts",
                               250, 0., 25.)

    n_generated = 0

    for i in range(n_events):

        if not pythia.next():
            continue
        n_generated += 1

        eventWeight = pythia.infoPython().weight()

        # ---- 1) N_ch in forward window ----
        n_ch = 0
        for j in range(pythia.event.size()):
            particle = pythia.event[j]
            if particle.isFinal() and particle.isCharged():
                eta = abs(particle.eta())
                if 2.8 < eta < 5.1:
                    n_ch += 1

        hist_nch.Fill(n_ch, eventWeight)

        # ---- 2) Protons/antiprotons in |eta| < 1 ----
        for j in range(pythia.event.size()):
            particle = pythia.event[j]
            if particle.isFinal() and abs(particle.id()) == 2212:
                if abs(particle.eta()) < 1.0:
                    pt = particle.pT()
                    hist_pt_proton.Fill(pt, eventWeight)
                    hist_pt_vs_nch.Fill(pt, n_ch, eventWeight)

        if i % 10000 == 0 and i > 0:
            print(f"  processed {i} requested / {n_generated} accepted")

    # ---- cross section AFTER generation ----
    pythia.stat()  # ensures sigmaGen/sigmaErr are up to date
    sigma_tot = pythia.infoPython().sigmaGen()
    sigma_err = pythia.infoPython().sigmaErr()
    print(f"\033[1;32m[INFO]\033[0m sigmaGen = {sigma_tot:.4e} +- {sigma_err:.2e} mb")
    print(f"\033[1;32m[INFO]\033[0m accepted events: {n_generated} / {n_events}")

    # Meta-info: store sigma and N_generated inside the same file
    hist_sigma = ROOT.TH1D("sigma_tot", "Total cross section [mb]", 1, 0., 1.)
    hist_sigma.SetBinContent(1, sigma_tot)
    hist_sigma.SetBinError(1, sigma_err)

    hist_n_gen = ROOT.TH1D("n_generated", "Number of generated events", 1, 0., 1.)
    hist_n_gen.SetBinContent(1, n_generated)

    hist_nch.Write()
    hist_pt_vs_nch.Write()
    hist_pt_proton.Write()
    hist_sigma.Write()
    hist_n_gen.Write()

    output_file.Close()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-n", "--nevents", type=int)
    parser.add_argument("-s", "--seed", type=int, default=1)

    args = parser.parse_args()

    if not args.input:
        print('\033[1;31mError:\033[0m input file must be specified; use option -i or --input')
        sys.exit(1)

    if not args.nevents:
        print('\033[1;31mError:\033[0m number of events must be specified; use option -n or --nevents')
        sys.exit(1)

    sys.exit(run_pythia(args.input, args.nevents, args.seed))
