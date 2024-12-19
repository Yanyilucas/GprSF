
from typing import Dict
import os
import sys
from model import viz

companies = ['NQUS500LC','COMP','OMHX25','NDX']
plotters:  Dict[str, viz.plotter] = {}
start_year = 2016


def main():
    init_data()
    for company in companies:
        make_summary(company)


def make_summary(company_name):
    plotter = plotters[company_name]

    plotter.show_whole_time_series() # 所有数据
    plotter.show_time_series(start_year=start_year, end_year=2024) # 限定年限数据
    plotter.show_preprocessed_prices(start_year=start_year, end_year=2024) #正则数据
    plotter.show_gp_prediction(train_start=start_year, train_end=2023, pred_year=2024) # 预测数据

    
    #@ train_start and train_end are the years used for training the model
    #@ 确保 pred_year 在 train_end 之后
    #plotter.show_gp_prediction(train_start=start_year, train_end=2022, pred_year=2024, pred_quarters= [3, 4])


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
