#!/usr/bin/env python3
"""
analyze_observables.py
"""

import os
import sys
import glob
from collections import OrderedDict

import ROOT
import numpy as np

# ============================================================
# CONFIG
# ============================================================

CENTRALITY = OrderedDict([
    ("pp",   [("MB", 0, 100)]),
    ("pPb",  [("0-20", 0, 20), ("20-40", 20, 40), ("40-100", 40, 100)]),
    ("PbPb", [("0-10", 0, 10), ("10-20", 10, 20), ("20-40", 20, 40),
              ("40-60", 40, 60), ("60-80", 60, 80)]),
])

NCOLL = {
    "pp":   {"MB": 1.0},
    "pPb":  {"0-20": 12.34, "20-40": 9.92, "40-100": 4.12},
    "PbPb": {"0-10": 1571.0, "10-20": 971.5, "20-40": 467.0,
             "40-60": 138.5, "60-80": 28.45},
}

DELTA_Y = 2.0
PT_MAX = 25.0
NCH_MAX = {"pp": 300.0, "pPb": 1500.0, "PbPb": 10000.0}
REBIN_FACTOR = 2 # Объединяем по 2 бина для подавления шума

OUTPUT_DIR = "output"
PICT_DIR = "pictures"
RESULTS_FILE = "output/analysis_results.root"

# ============================================================
# I/O
# ============================================================

def find_seed_files(system):
    pattern = os.path.join(OUTPUT_DIR, f"{system}5020_seed*.root")
    return [f for f in sorted(glob.glob(pattern)) if os.path.getsize(f) > 1024]

def load_and_merge(system):
    files = find_seed_files(system)
    if not files:
        raise RuntimeError(f"No files for {system}")
    print(f"  {system}: merging {len(files)} file(s)")

    hnch = h2d = None
    sigma_w = 0.0
    n_total = 0

    for fp in files:
        f = ROOT.TFile.Open(fp)
        if not f or f.IsZombie():
            continue
        n_gen = f.Get("n_generated").GetBinContent(1)
        sigma = f.Get("sigma_tot").GetBinContent(1)
        if n_gen <= 0:
            f.Close()
            continue

        h1 = f.Get("N_ch"); h1.SetDirectory(0)
        h2 = f.Get("pT_vs_Nch"); h2.SetDirectory(0)
        if hnch is None:
            hnch = h1.Clone(f"{system}_Nch"); hnch.SetDirectory(0)
            h2d = h2.Clone(f"{system}_ptNch"); h2d.SetDirectory(0)
        else:
            hnch.Add(h1); h2d.Add(h2)
        sigma_w += sigma * n_gen
        n_total += n_gen
        f.Close()

    if n_total == 0:
        raise RuntimeError(f"No valid events for {system}")
    sigma = sigma_w / n_total
    print(f"    events = {n_total:.0f}, sigma = {sigma:.3e} mb")
    return hnch, h2d, sigma, n_total


# ============================================================
# Centrality
# ============================================================

def find_nch_boundaries(hnch, classes):
    total = hnch.Integral()
    if total <= 0:
        raise RuntimeError("empty N_ch")
    n_bins = hnch.GetNbinsX()
    cum = {hnch.GetXaxis().GetBinLowEdge(n_bins + 1): 0.0}
    acc = 0.0
    for i in range(n_bins, 0, -1):
        acc += hnch.GetBinContent(i)
        cum[hnch.GetXaxis().GetBinLowEdge(i)] = acc / total

    def nch_at_pct(pct):
        target = pct / 100.0
        for nch in sorted(cum.keys(), reverse=True):
            if cum[nch] >= target:
                return nch
        return min(cum.keys())
    return {label: (nch_at_pct(hi), nch_at_pct(lo)) for label, lo, hi in classes}


def n_events_in(hnch, nmin, nmax):
    s = 0.0
    for i in range(1, hnch.GetNbinsX() + 1):
        x = hnch.GetXaxis().GetBinCenter(i)
        if nmin <= x < nmax:
            s += hnch.GetBinContent(i)
    return s


# ============================================================
# Spectrum
# ============================================================

def project_pt(h2d, nmin, nmax):
    ya = h2d.GetYaxis()
    bmin = ya.FindBin(nmin)
    bmax = ya.FindBin(nmax - 1e-6)
    if bmax < bmin:
        return None
    h = h2d.ProjectionX(f"hpt_{nmin:.0f}_{nmax:.0f}", bmin, bmax)
    h.SetDirectory(0)
    return h


def compute_yield(h_pt, n_events):
    n = h_pt.GetNbinsX()
    out = ROOT.TH1D(h_pt.GetName() + "_yield", "", n, h_pt.GetXaxis().GetXmin(), h_pt.GetXaxis().GetXmax())
    out.SetDirectory(0)
    for i in range(1, n + 1):
        pt = h_pt.GetXaxis().GetBinCenter(i)
        if pt <= 0 or n_events <= 0:
            continue
        dpt = h_pt.GetXaxis().GetBinWidth(i)
        f = 1.0 / (n_events * 2 * np.pi * pt * dpt * DELTA_Y)
        out.SetBinContent(i, h_pt.GetBinContent(i) * f)
        out.SetBinError(i, h_pt.GetBinError(i) * f)
    return out


def yield_to_xsec(h_yield, sigma):
    out = h_yield.Clone(h_yield.GetName() + "_xsec")
    out.SetDirectory(0)
    out.Scale(sigma)
    return out


# ============================================================
# Ratios
# ============================================================

def ratio_R_AB(y_AB, ncoll_AB, y_pp):
    n = y_AB.GetNbinsX()
    out = ROOT.TH1D("R_AB", "", n, y_AB.GetXaxis().GetXmin(), y_AB.GetXaxis().GetXmax()); out.SetDirectory(0)
    for i in range(1, n + 1):
        a = y_AB.GetBinContent(i); p = y_pp.GetBinContent(i)
        ea = y_AB.GetBinError(i);   ep = y_pp.GetBinError(i)
        if p <= 0 or a <= 0:
            continue
        val = a / (ncoll_AB * p)
        rel = (ea/a)**2 + (ep/p)**2
        out.SetBinContent(i, val)
        out.SetBinError(i, val * np.sqrt(rel) if rel > 0 else 0)
    return out


def ratio_R_CP(y_c, ncoll_c, y_p, ncoll_p):
    n = y_c.GetNbinsX()
    out = ROOT.TH1D("R_CP", "", n, y_c.GetXaxis().GetXmin(), y_c.GetXaxis().GetXmax()); out.SetDirectory(0)
    for i in range(1, n + 1):
        c = y_c.GetBinContent(i); p = y_p.GetBinContent(i)
        ec = y_c.GetBinError(i);   ep = y_p.GetBinError(i)
        if c <= 0 or p <= 0:
            continue
        val = (c / ncoll_c) / (p / ncoll_p)
        rel = (ec/c)**2 + (ep/p)**2
        out.SetBinContent(i, val)
        out.SetBinError(i, val * np.sqrt(rel) if rel > 0 else 0)
    return out


# ============================================================
# Plotting
# ============================================================

COLORS = [ROOT.kRed+1, ROOT.kBlue+1, ROOT.kGreen+2,
          ROOT.kOrange+7, ROOT.kMagenta+1, ROOT.kCyan+2]
MARKERS = [20, 21, 22, 23, 33, 34]


def setup_style():
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(False)
    ROOT.gStyle.SetOptTitle(False)


def new_canvas(name="c"):
    c = ROOT.TCanvas(name, "", 900, 800)
    ROOT.gPad.SetLeftMargin(0.15)
    ROOT.gPad.SetRightMargin(0.05)
    ROOT.gPad.SetTopMargin(0.05)
    ROOT.gPad.SetBottomMargin(0.15)
    return c


def get_legend(pos):
    # pos: "NW" (Top-Left), "SW" (Bottom-Left), "SE" (Bottom-Right)
    if pos == "NW":
        leg = ROOT.TLegend(0.15, 0.75, 0.45, 0.93)
    elif pos == "SW":
        leg = ROOT.TLegend(0.15, 0.15, 0.45, 0.35)
    elif pos == "SE":
        leg = ROOT.TLegend(0.70, 0.15, 0.93, 0.35)
    else:
        leg = ROOT.TLegend(0.65, 0.72, 0.93, 0.93)
    leg.SetLineColorAlpha(ROOT.kGray, 0.5)
    leg.SetFillColorAlpha(ROOT.kWhite, 0.7)
    leg.SetBorderSize(1)
    return leg

#first.GetYaxis().SetTitle("#frac{1}{2#pi p_{T}} #frac{d^{2}#sigma}{dp_{T}dy} [mb/(GeV/c)^{2}]")
def plot_spectrum(system, spectra):
    c = new_canvas(f"c_spectrum_{system}")
    c.SetLogy()
    leg = get_legend("NE")
    first = None
    for idx, (label, h) in enumerate(spectra.items()):
        h.SetLineColor(COLORS[idx % len(COLORS)])
        h.SetLineWidth(2)
        h.SetLineStyle(idx + 1 if idx > 0 else 1)
        if first is None:
            first = h
    if first is None:
        return

    # Оформление осей (используем красивую букву S вместо длинной формулы)
    first.GetXaxis().SetTitle("p_{T} [GeV/c]")
    first.GetYaxis().SetTitle("#frac{1}{2#pi p_{T}} #frac{d^{2}#sigma}{dp_{T}dy} [mb/(GeV/c)^{2}]")
    first.GetXaxis().SetTitleSize(0.045)
    first.GetYaxis().SetTitleSize(0.045)
    first.GetXaxis().SetTitleOffset(1.2)
    first.GetYaxis().SetTitleOffset(1.4) # Чуть сдвинул влево, чтобы не прилипало

    first.GetXaxis().SetRangeUser(0.0, 12.0)
    first.SetMinimum(1e-5)
    first.SetMaximum(first.GetMaximum() * 5)

    c.cd()
    first.Draw("HIST")
    for label, h in list(spectra.items())[1:]:
        h.Draw("HIST SAME")
    for label, h in spectra.items():
        leg.AddEntry(h, label, "L")
    leg.Draw()
    os.makedirs(PICT_DIR, exist_ok=True)
    c.SaveAs(f"{PICT_DIR}/spectrum_{system}.pdf")
    c.SaveAs(f"{PICT_DIR}/spectrum_{system}.png")
    print(f"  -> {PICT_DIR}/spectrum_{system}.pdf")


def plot_ratio(title, ratios, fname, legend_pos="NE"):
    c = new_canvas(f"c_ratio_{fname}")
    leg = get_legend(legend_pos)
    first = None
    for idx, (label, h) in enumerate(ratios.items()):
        h.SetLineColor(COLORS[idx % len(COLORS)])
        h.SetMarkerColor(COLORS[idx % len(COLORS)])
        h.SetMarkerStyle(MARKERS[idx % len(MARKERS)])
        h.SetMarkerSize(1.2)
        h.SetLineWidth(2)
        if first is None:
            first = h
    if first is None:
        return

    # Оформление осей
    first.GetXaxis().SetTitle("p_{T} [GeV/c]")
    first.GetYaxis().SetTitle(title)
    first.GetXaxis().SetTitleSize(0.045)
    first.GetYaxis().SetTitleSize(0.045)
    first.GetXaxis().SetTitleOffset(1.2)
    first.GetYaxis().SetTitleOffset(1.4)

    first.GetXaxis().SetRangeUser(0.0, 8.0)
    first.SetMinimum(0.0)
    first.SetMaximum(2.0)

    c.cd()
    first.Draw("AXIS")
    first.Draw("SAME AXIS X+ Y+")  # <--- ВОТ ОН, СПАСИТЕЛЬНЫЙ SAME!

    line = ROOT.TLine(0, 1, 8.0, 1)
    line.SetLineStyle(2); line.SetLineColor(ROOT.kGray+2)
    line.Draw()

    for label, h in ratios.items():
        leg.AddEntry(h, label, "LP")
        h.Draw("SAME E")
    leg.Draw()
    os.makedirs(PICT_DIR, exist_ok=True)
    c.SaveAs(f"{PICT_DIR}/{fname}.pdf")
    c.SaveAs(f"{PICT_DIR}/{fname}.png")
    print(f"  -> {PICT_DIR}/{fname}.pdf")


# ============================================================
# Main
# ============================================================

def main():
    setup_style()
    os.makedirs(PICT_DIR, exist_ok=True)

    print("=" * 60); print("Loading"); print("=" * 60)
    data = {}
    for system in CENTRALITY:
        try:
            hnch, h2d, sigma, ngen = load_and_merge(system)
            data[system] = dict(hnch=hnch, h2d=h2d, sigma=sigma, ngen=ngen)
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            if system == "pp":
                sys.exit(1)

    # pp reference (с применением Rebin)
    pp = data["pp"]
    pp_pt = project_pt(pp["h2d"], 0.0, NCH_MAX["pp"] + 1.0)
    pp_pt = pp_pt.Rebin(REBIN_FACTOR) # Уменьшаем шум
    pp_yield = compute_yield(pp_pt, pp["hnch"].Integral())
    saved_results = {"pp": {"MB": yield_to_xsec(pp_yield, pp["sigma"])}}

    for system in ["pPb", "PbPb"]:
        if system not in data:
            continue
        d = data[system]
        hnch, h2d, sigma = d["hnch"], d["h2d"], d["sigma"]
        print("=" * 60); print(f"System {system}"); print("=" * 60)

        bounds = find_nch_boundaries(hnch, CENTRALITY[system])
        for label, (nmin, nmax) in bounds.items():
            nev = n_events_in(hnch, nmin, nmax)
            print(f"  {label:8s}  N_ch in [{nmin:.0f}, {nmax:.0f}],  events = {nev:.0f}")

        spectra = OrderedDict()
        yields = OrderedDict()
        for label, (nmin, nmax) in bounds.items():
            h_pt = project_pt(h2d, nmin, nmax)
            if h_pt is None:
                continue
            h_pt = h_pt.Rebin(REBIN_FACTOR) # Уменьшаем шум

            nev = n_events_in(hnch, nmin, nmax)
            if nev <= 0:
                continue
            y = compute_yield(h_pt, nev)
            yields[label] = y
            spectra[label] = yield_to_xsec(y, sigma)

        plot_spectrum(system, spectra)
        saved_results[system] = spectra

        # Строим R_AB и переносим легенду на северо-запад (NW)
        rabs = OrderedDict()
        for label, y in yields.items():
            rabs[label] = ratio_R_AB(y, NCOLL[system][label], pp_yield)
        plot_ratio(f"#font[12]{{R_{{AB}}}}", rabs, f"R_AB_{system}", legend_pos="NW")

        # Строим R_CP
        periph = list(yields.keys())[-1]
        rcps = OrderedDict()
        for label, y in yields.items():
            if label == periph:
                continue
            rcps[f"{label} / {periph}"] = ratio_R_CP(
                y, NCOLL[system][label],
                yields[periph], NCOLL[system][periph],
            )

        leg_pos = "SW" if system == "pPb" else "NW"
        plot_ratio(f"#font[12]{{R_{{CP}}}}", rcps, f"R_CP_{system}", legend_pos=leg_pos)

    print("=" * 60); print(f"Saving {RESULTS_FILE}"); print("=" * 60)
    rf = ROOT.TFile(RESULTS_FILE, "RECREATE")
    for system, spectra in saved_results.items():
        for label, h in spectra.items():
            safe = label.replace(" ", "_").replace("/", "_").replace("%", "pct")
            h.Write(f"spectrum_{system}_{safe}")
    rf.Close()
    print("Done. Plots in pictures/")

if __name__ == "__main__":
    main()
