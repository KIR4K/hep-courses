#!/usr/bin/env python3

import os
import sys
import argparse

import ROOT

import pythia8

def get_system_name(cmnd_file_path: str) -> str:
    """
    Читает .cmnd файл и ищет строку вида '! System: <name>'.
    Возвращает <name> (например, 'pp5020', 'pPb5020', 'PbPb5020').
    Если строка не найдена, возвращает 'unknown'.
    """
    with open(cmnd_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("! System:"):
                return line.split(":", 1)[1].strip()
    return "unknown"

def run_pythia(input_file_name : str, n_events : int) -> int :

    # checking if specified input file exists
    if not os.path.isfile(input_file_name) :
        print(f"\033[1;31mError:\033[0m file {input_file_name} was not found")
        return 1

    # creating and configuring Pythia instance
    # full manual is avaliable at https://pythia8.web.cern.ch/manuals/pythia8317/
    # (switch the version if you have a newer one)
    pythia = pythia8.Pythia()

    # random generator seed; main can be changed to have seed argument. Seed is needed if you intend to perform calculations on multiple threads via bash scripts
    seed = 1;
    # setting the random seed
    pythia.readString("Random:seed = " + str(seed))

    # To do: determine whether HardQCD and/or SoftQCD needed in this analysis
    # Hint: look at SoftQCD:nonDiffractive definition (remind: you perform a minimum bias study)
    # (HardQCD: https://pythia.org/latest-manual/QCDHardProcesses.html)
    # (SoftQCD: https://pythia8.web.cern.ch/manuals/pythia8317/QCDSoftProcesses.html)
    # pythia.readString("HardQCD:all = on")
    pythia.readString("SoftQCD:nonDiffractive = on")
    # (Collision and beam parameters: https://pythia.org/latest-manual/BeamParameters.html)
    # (Phase space cuts: https://pythia.org/latest-manual/PhaseSpaceCuts.html)
    # (PDF selection: https://pythia.org/latest-manual/PDFSelection.html)
    # setting parameters from .cmnd input file
    pythia.readFile(input_file_name)

    # uncomment the next line if you do not need for pythia to print event info and banner
    # pythia.readString("Print:quiet = on")

    # initializing pythia; this step pythia applies parameters we set earlier and
    # checks if there are problems with the current specification
        # initializing pythia
    pythia.init()

    sigma_tot = pythia.infoPython().sigmaGen()

    print(f"\033[1;32m[INFO]\033[0m Total cross section: {sigma_tot:.4f} mb")

    # creating directory in which the file outputFile will be written
    os.makedirs("output", exist_ok=True)

    # Получаем уникальное имя системы из .cmnd файла
    system_name = get_system_name(input_file_name)
    print(f"\033[1;32m[INFO]\033[0m Detected system: {system_name}")

    output_file = ROOT.TFile(f"output/{system_name}.root", "RECREATE")
    # if another file is created after this with option "RECREATE", "UPDATE", or "CREATE"
    # root will try to write TObject objects to the new defined file
    # if you have multiple TFile files you write in in yor program use
    # TFile::cd() to point root to the file you need

    # To do: declare histograms to store number of event charged particles multiplicity
    # and pT multiplicity vs number of charged particles

    # 1D гистограмма для множественности заряженных частиц (N_ch) в forward-области
    # Диапазон 0-300 подобран с запасом для pp, для PbPb может потребоваться больше
    hist_nch = ROOT.TH1D("N_ch", ";N_{ch};Events", 300, 0., 300.)

    # 2D гистограмма: pT частицы vs N_ch события
    # Ось X - pT (0-25 ГэВ), Ось Y - N_ch (0-300)
    hist_pt_vs_nch = ROOT.TH2D("pT_vs_Nch", ";p_{T} [GeV/c];N_{ch}",
                            250, 0., 25.,
                            300, 0., 300.)

    # 1D гистограмма для спектра pT протонов/антипротонов (для варианта 4)
    hist_pt_proton = ROOT.TH1D("pT_proton", ";p_{T} [GeV/c];Counts",
                           250, 0., 25.)
    # example: pT multiplicity
    hist_pt = ROOT.TH1D("pT multiplicity", "", 200, 0., 25.)

    # Получаем полное сечение (в мб) после инициализации для последующей нормировки
    #sigma_tot = pythia.infoPython().sigmaGen()

    # Сохраняем сечение в файл как однобинную гистограмму для удобного чтения в analyze_output.py
    hist_sigma = ROOT.TH1D("sigma_tot", "Total cross section [mb]", 1, 0., 1.)
    hist_sigma.SetBinContent(1, sigma_tot)
    hist_sigma.Write()

    # Итерируемся по событиям
    for i in range(n_events):
        if not pythia.next():
            continue

        eventWeight = pythia.infoPython().weight()

        # 1. Считаем N_ch в forward-области (2.8 < |eta| < 5.1) для определения центральности
        n_ch = 0
        for j in range(pythia.event.size()):
            particle = pythia.event[j]
            # Проверяем, что частица конечная и заряженная
            if particle.isFinal() and particle.isCharged():
                eta = abs(particle.eta())
                if 2.8 < eta < 5.1:
                    n_ch += 1

        # Заполняем 1D гистограмму множественности для этого события
        hist_nch.Fill(n_ch, eventWeight)

        # 2. Ищем протоны и антипротоны (Вариант 4) в центральной области (|eta| < 1)
        for j in range(pythia.event.size()):
            particle = pythia.event[j]
            pid = particle.id()

            # Проверяем, что это протон (2212) или антипротон (-2212), и что она конечная
            if particle.isFinal() and abs(pid) == 2212:
                eta = abs(particle.eta())
                if eta < 1.0:
                    pt = particle.pT()
                    # Заполняем 1D спектр (сумма p и p~)
                    hist_pt_proton.Fill(pt, eventWeight)
                    # Заполняем 2D гистограмму: pT (ось X) vs N_ch (ось Y)
                    hist_pt_vs_nch.Fill(pt, n_ch, eventWeight)

        # Печатаем прогресс раз в 1000 событий, чтобы не спамить в консоль
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i} events.")

    # Масштабируем гистограмму протонов на 1/2, так как по заданию нужно (p + p~)/2
    hist_pt_proton.Scale(0.5)

    # Записываем все гистограммы в файл
    hist_nch.Write()
    hist_pt_vs_nch.Write()
    hist_pt_proton.Write()
    # hist_pt.Write() # Можно оставить раскомментированным для отладки, если нужен спектр всех частиц

    output_file.Close()
    return 0

if __name__ == "__main__" :
    # parsing input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-n", "--nevents", type=int)

    args = parser.parse_args()
    input_file_name = args.input
    n_events = args.nevents

    # requiring for the input file to be passed
    if not input_file_name :
        print('\033[1;31mError:\033[0m input file must be specified; use option -i or --input')
        sys(exit(1))

    # requiring for the number of events to be passed
    if not n_events :
        print('\033[1;31mError:\033[0m number of events must be specified; use option -n or --nevents')
        sys(exit(1))

    sys.exit(run_pythia(input_file_name, n_events))
