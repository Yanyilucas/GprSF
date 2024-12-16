
from typing import Dict
import os
import sys
from model import viz

companies = []
plotters:  Dict[str, viz.plotter] = {}
start_year = 2015


def main():
    init_data()
    for company in companies:
        make_summary(company)


def make_summary(company_name):
    plotter = plotters[company_name]

    plotter.show_whole_time_series()
    plotter.show_time_series(start_year=start_year, end_year=2016)
    plotter.show_preprocessed_prices(start_year=start_year, end_year=2016)
    plotter.show_gp_prediction(train_start=start_year, train_end=2016, pred_year=2017)
    plotter.show_time_series(start_year=start_year, end_year=2018)
    plotter.show_gp_prediction(train_start=start_year, train_end=2018, pred_year=2018, pred_quarters= [3, 4])


def init_data():
    if companies == []:
        for company in os.listdir('data'):
            current_company = company.split('.')[0]
            companies.append(current_company)
            plotters[current_company] = (viz.plotter(company_name=current_company))
    else:
        for company in companies:
            plotters[company] = (viz.plotter(company_name=company))

if __name__ == "__main__":
    main()
