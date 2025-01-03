import numpy as np
import pandas as pd
import torch
import gpytorch
from tqdm import tqdm
from . import dataset 


class ExactGPModel(gpytorch.models.ExactGP):
    """
    定义一个基于 GPyTorch 的高斯过程回归模型。
    """
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        # __import__('ipdb').set_trace()
        self.mean_module = gpytorch.means.ConstantMean()
        
        # 卷积核函数：复合核函数，结合了不同的核函数
        self.covar_module = gpytorch.kernels.ScaleKernel(
            # Matern 核，适用于波动性
            gpytorch.kernels.MaternKernel(nu=2.5, lengthscale=0.005) 
            + 
            # RBF 核，适用于平滑的长期趋势
            gpytorch.kernels.RBFKernel(lengthscale=0.1)
        )


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPR:
    """
    使用 GPyTorch 实现的高斯过程回归模型封装类。
    """
    def __init__(self, company_name: str):
        self.company_data = dataset.csv_handler(company_name)
        self.prices_data = self.company_data.get_equal_length_prices()#升序时间顺序
        self.quarters = self.company_data.quarters
        self.years = self.company_data.years
        self.max_days = self.company_data.max_days

        self.model = None
        self.likelihood = None
        self.kernels = []

        self.iterations = 120

    def get_eval_model(self, start_year: int, end_year: int, pred_year: int, pred_quarters: list = None):
        """
        训练高斯过程回归模型并进行预测，使用 GPyTorch。
        :param start_year: 训练数据起始年份
        :param end_year: 训练数据结束年份
        :param pred_year: 预测年份
        :param pred_quarters: 预测季度列表
        :return: (x_mesh, y_mean, y_cov)
        """

        years_quarters = list(range(start_year, end_year + 1)) + ['Quarter']
        training_years = years_quarters[:-2]
        df_prices = self.prices_data[self.prices_data.columns.intersection(years_quarters)]#!升序的


        possible_days = list(df_prices.index.values)
        
        # first year prices
        first_year_prices = df_prices[start_year]
        if start_year == self.company_data.years[0]:
            first_year_prices = first_year_prices[first_year_prices != 0]
            first_year_prices = pd.concat([
                pd.Series([0.0], index=[first_year_prices.index[0] - 1]),
                first_year_prices
            ])

        data_X = []
        data_Y = []

        first_year_days = first_year_prices.index.values
        for day in first_year_days:
            data_X.append([start_year, day])
        data_Y.extend(first_year_prices.values)

        # other years prices
        for current_year in training_years[1:]:
            current_year_prices = df_prices[current_year].dropna().values
            current_year_days = possible_days[:len(current_year_prices)]
            for day in current_year_days:
                data_X.append([current_year, day])
            data_Y.extend(current_year_prices)

        # last year prices
        last_year_prices = df_prices[end_year].dropna()
        if pred_quarters is not None:
            length = 63 * (pred_quarters[0] - 1)
            last_year_prices = last_year_prices.iloc[:length]
        last_year_days = last_year_prices.index.values
        for day in last_year_days:
            data_X.append([end_year, day])
        data_Y.extend(last_year_prices.values)

        X = np.array(data_X)
        Y = np.array(data_Y)

        # predict days
        if pred_quarters is not None:
            pred_days = list(range(63 * (pred_quarters[0] - 1), 63 * pred_quarters[-1]))
        else:
            pred_days = list(range(0, self.max_days))

        x_mesh = np.linspace(pred_days[0], pred_days[-1], 2000)
        x_pred = np.column_stack((np.full_like(x_mesh, pred_year), x_mesh))

        # 构建 GPyTorch 模型和似然函数
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        train_x = torch.from_numpy(X).float()
        train_y = torch.from_numpy(Y).float()

        self.model = ExactGPModel(train_x, train_y, self.likelihood)


        self.model.train()
        self.likelihood.train()


        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},  
        ], lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Training
        for _ in tqdm(range(self.iterations), desc="Training GPyTorch GP", unit="iter"):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y) # 负对数边际似然
            print('Iter %d/%d - Loss: %.3f' % (_, self.iterations, loss.item()))
            loss.backward()
            optimizer.step()


        # eval
        self.model.eval()
        self.likelihood.eval()

        test_x = torch.from_numpy(x_pred).float()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(test_x))
            y_mean = observed_pred.mean.numpy()
            y_cov = observed_pred.variance.numpy()

        return x_mesh, y_mean, y_cov
