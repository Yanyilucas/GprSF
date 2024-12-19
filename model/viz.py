import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os


from . import dataset 
from . import GaussianProcess as GaussianProcess

class plotter:
    company_name = None
    company_handler = None
    prices_data = None
    quarters = None
    years = None
    max_days = None
    quarter_length = None
    gpr = None
    picdir="./pic"
    def __init__(self, company_name: str):
        self.isGpytorch = True
        self.company_name = company_name
        self.company_handler = dataset.csv_handler(company_name)
        self.prices_data = self.company_handler.get_equal_length_prices()#! 升序
        self.quarters = self.company_handler.quarters
        self.years = self.company_handler.years
        # print(self.years)
        self.max_days = self.company_handler.max_days
        self.quarter_length = int(self.max_days / 4)
        if self.isGpytorch:
            self.gpr = GaussianProcess.GPR(company_name)
        else:
            raise NotImplementedError("Not implemented for GPy")
        self.company_dir=os.path.join(self.picdir,self.company_name)

        os.makedirs(self.company_dir, exist_ok=True)
        
    def show_preprocessed_price(self, year: int):
        self.show_preprocessed_prices(start_year=year, end_year=year)

    def show_preprocessed_prices(self, start_year: int, end_year: int):
        self.validate_dates(start_year=start_year, end_year=end_year)

        fig = plt.figure(num=self.company_name + ' normalized prices')
        ax = plt.gca()
        fig.set_size_inches(12, 6)
        y_low, y_high = 0, 0
        for year in range(start_year, end_year + 1):
            y = self.prices_data[year]
            #__import__('ipdb').set_trace()
            if y_low >= min(y):
                y_low = min(y)
            if y_high <= max(y):
                y_high = max(y)
            x = np.linspace(0, len(y), len(y))
            plt.plot(x, y, alpha=.95, label=year)
            plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

        y_max = max(abs(y_low) - 1, abs(y_high) + 1)
        x_min, x_max = -10, self.max_days + 10
        ax.set_ylim(bottom=-y_max, top=y_max)
        ax.set_xlim(left=x_min, right=x_max)

        for i in range(0, 5):
            plt.vlines(x=(self.max_days / 4) * i, ymin=-y_max, ymax=y_max, color='black', linestyles='--', alpha=.6,
                       zorder=-1)
            if i < 4:
                ax.text((self.max_days / 4) * i + self.max_days / 8 - 5, y_max - 0.5, self.quarters[i],
                        fontsize=12)
                
        plt.hlines(y=0, xmin=x_min, xmax=x_max, color='black', linestyles='--', alpha=.6, zorder=-1)

        plt.grid(True, alpha=.25)
        plt.title(self.company_name)
        plt.xlabel('Days')
        plt.ylabel('Normalized price')

        plt.tight_layout()
        
        fname = '{}-{}-normalized_prices.png'.format(start_year, end_year)
        fname = os.path.join(self.company_dir, fname)
        fig.savefig(fname, dpi=fig.dpi)
        plt.close(fig)  # 确保释放图像对象，避免复用

    def show_gp_prediction(self, train_start: int, train_end: int, pred_year: int, pred_quarters: list = None):
        """
        show prediction diagram
        """
        #__import__('ipdb').set_trace()
        self.validate_dates(start_year=train_start, end_year=pred_year)

        prices = self.prices_data[pred_year]
        prices = prices[prices.iloc[:].notnull()]

        fig = plt.figure(num=self.company_name + ' prediction')
        ax = plt.gca()
        fig.set_size_inches(12, 6)

        x_obs = list(range(prices.index[0], prices.index[-1] + 1))
        x_mesh, y_mean, y_cov = self.gpr.get_eval_model(start_year=train_start, end_year=train_end,
                                                          pred_year=pred_year,
                                                          pred_quarters=pred_quarters)
        if self.isGpytorch:
            #* [μ-σ, μ+σ] 
            #注意 y_cov的值是不一样的,但在数值上接近
            y_lower = y_mean - np.sqrt(y_cov)
            y_upper = y_mean + np.sqrt(y_cov)
        else:    
            raise NotImplementedError("Not implemented for GPy")

        y_max = max(abs(min(y_lower) - 1), abs(max(y_upper) + 1))
        ax.set_ylim(bottom=-y_max, top=y_max)

        x_min, x_max = -10, self.max_days + 10
        ax.set_xlim(left=x_min, right=x_max)

        plt.plot(x_obs, prices, color='#006699', alpha=.95, label=u'Observations ' + str(pred_year), zorder=10)
        plt.plot(x_mesh, y_mean, color='#ff0066', linestyle='--', label=u'Prediction')
        plt.fill_between(x_mesh, y_lower, y_upper,
                         alpha=.25, label='[μ-σ, μ+σ]', color='#ff0066')

        handles, labels = plt.gca().get_legend_handles_labels()
        new_labels, new_handles = [], []
        for handle, label in zip(handles, labels):
            if label not in new_labels:
                new_labels.append(label)
                new_handles.append(handle)
        plt.legend(new_handles, new_labels, bbox_to_anchor=(0.01, 0.02), loc='lower left', borderaxespad=0.)

        for i in range(0, 5):
            plt.vlines(x=self.quarter_length * i, ymin=-y_max, ymax=y_max, color='black', linestyles='--', alpha=.6,
                       zorder=-1)
            if i < 4:
                ax.text(self.quarter_length * i + self.quarter_length / 2 - 5, y_max - 0.5, self.quarters[i],
                        fontsize=12)
        plt.hlines(y=0, xmin=x_min, xmax=x_max, color='black', linestyles='--', alpha=.6, zorder=-1)

        plt.grid(True, alpha=.25)
        plt.title(self.company_name)
        plt.xlabel('Days\n')
        plt.ylabel('Normalized price')

        plt.tight_layout()

        
        fname = '{}-prediction.png'.format(pred_year)
        fname = os.path.join(self.company_dir, fname)
        fig.savefig(fname, dpi=fig.dpi)
        plt.close(fig)  # 确保释放图像对象，避免复用

    def show_whole_time_series(self, intermediate: bool = False):
        self.show_time_series(start_year=self.years[0], end_year=self.years[-1], intermediate=intermediate)

    def show_time_series(self, start_year: int, end_year: int, intermediate: bool = True):
        self.validate_dates(start_year=start_year, end_year=end_year)

        prices_data = self.company_handler.get_whole_prices(start_year=start_year, end_year=end_year)

        fig = plt.figure(num=self.company_name + ' prices')
        fig.set_size_inches(12, 6)
        plt.plot(prices_data.iloc[:, 0], prices_data.iloc[:, 1], color='#006699', alpha=.95,
                 label=u'Observations ' + str(start_year) + '-' + str(end_year), zorder=10)
        ax = plt.gca()

        x_ticks = []
        for year in range(start_year, end_year + 2):
            if year == end_year + 1:
                current_date = prices_data[prices_data['Date'].dt.year == end_year].iloc[-1, 0]
            else:
                current_date = prices_data[prices_data['Date'].dt.year == year].iloc[0, 0]
            x_ticks.append(current_date)

        x_formatter = mdates.DateFormatter('%d-%m-%Y')
        ax.xaxis.set_major_formatter(x_formatter)
        if not intermediate:
            x_ticks = [x_ticks[0], x_ticks[-2], x_ticks[-1]]
            ax.set_xticks([x_ticks[0], x_ticks[-1]])
        else:
            ax.set_xticks(x_ticks)
        plt.xticks(rotation=20)
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        ax.set_ylim(bottom=y_min, top=y_max)
        ax.set_xlim(left=x_min, right=x_max)

        for i in range(0, len(x_ticks)):
            plt.vlines(x=x_ticks[i], ymin=y_min, ymax=y_max, color='black', linestyles='--', alpha=.6,
                       zorder=-1)

        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.title(self.company_name)
        plt.ylabel('Price')

        plt.tight_layout()

        fname = '{}-{}-prices.png'.format(start_year, end_year)
        fname = os.path.join(self.company_dir, fname)
        fig.savefig(fname, dpi=fig.dpi)
        plt.close(fig)  # 确保释放图像对象，避免复用

    def validate_dates(self, start_year: int, end_year: int):
        if start_year < self.years[0] or end_year > self.years[-1]:
            
            raise ValueError('\n' +
                             'Input years out of available range! \n' +
                             'Max range available: {}-{}\n'.format(self.years[0], self.years[-1]) +
                             'Was: {}-{}'.format(start_year, end_year))
