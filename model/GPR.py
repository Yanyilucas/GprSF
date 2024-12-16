import numpy as np
import pandas as pd
import torch
import gpytorch
from tqdm import tqdm
from . import data_handler 


class ExactGPModel(gpytorch.models.ExactGP):
    """
    定义一个基于 GPyTorch 的高斯过程回归模型。
    """
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        # 均值函数
        self.mean_module = gpytorch.means.ConstantMean()
        # 卷积核函数，ScaleKernel 包裹 RBFKernel
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class WrapperGPyTorch:
    """
    使用 GPyTorch 实现的高斯过程回归模型封装类。
    """
    def __init__(self, company_name: str):
        self.__company_data = data_handler.csv_handler(company_name)
        self.__prices_data = self.__company_data.get_equal_length_prices()
        self.__quarters = self.__company_data.quarters
        self.__years = self.__company_data.years
        self.__max_days = self.__company_data.max_days

        # GPyTorch 相关属性
        self.__model = None
        self.__likelihood = None
        self.__kernels = []

        # 训练迭代次数，可以根据需求调整
        self.__iterations = 50

    def get_eval_model(self, start_year: int, end_year: int, pred_year: int, pred_quarters: list = None):
        """
        训练高斯过程回归模型并进行预测，使用 GPyTorch。
        :param start_year: 训练数据起始年份
        :param end_year: 训练数据结束年份
        :param pred_year: 预测年份
        :param pred_quarters: 预测季度列表
        :return: (x_mesh, y_mean, y_cov)
        """
        # 1. 准备训练数据 X, Y
        years_quarters = list(range(start_year, end_year + 1)) + ['Quarter']
        training_years = years_quarters[:-2]
        df_prices = self.__prices_data[self.__prices_data.columns.intersection(years_quarters)]

        possible_days = list(df_prices.index.values)
        
        # 处理第一个年份数据
        first_year_prices = df_prices[start_year]
        if start_year == self.__company_data.years[0]:
            first_year_prices = first_year_prices[first_year_prices != 0]
            first_year_prices = pd.concat([
                pd.Series([0.0], index=[first_year_prices.index[0] - 1]),
                first_year_prices
            ])

        # 收集所有训练数据
        data_X = []
        data_Y = []

        # 已经处理好的 first_year_prices
        first_year_days = first_year_prices.index.values
        for day in first_year_days:
            data_X.append([start_year, day])
        data_Y.extend(first_year_prices.values)

        # 处理剩余年份
        for current_year in training_years[1:]:
            current_year_prices = df_prices[current_year].dropna().values
            current_year_days = possible_days[:len(current_year_prices)]
            for day in current_year_days:
                data_X.append([current_year, day])
            data_Y.extend(current_year_prices)

        # 处理最后一年的数据
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

        # 2. 准备预测数据 x_pred
        if pred_quarters is not None:
            pred_days = list(range(63 * (pred_quarters[0] - 1), 63 * pred_quarters[-1]))
        else:
            pred_days = list(range(0, self.__max_days))

        x_mesh = np.linspace(pred_days[0], pred_days[-1], 2000)
        x_pred = np.column_stack((np.full_like(x_mesh, pred_year), x_mesh))

        # 3. 构建 GPyTorch 模型和似然函数
        self.__likelihood = gpytorch.likelihoods.GaussianLikelihood()
        train_x = torch.from_numpy(X).float()
        train_y = torch.from_numpy(Y).float()

        self.__model = ExactGPModel(train_x, train_y, self.__likelihood)

        # 4. 训练模型，使用 tqdm 显示进度
        self.__model.train()
        self.__likelihood.train()

        optimizer = torch.optim.Adam([
        {'params': self.__model.mean_module.parameters(), 'lr': 0.1},  # 仅包含模型均值部分参数
        {'params': self.__model.covar_module.parameters(), 'lr': 0.1}, # 仅包含模型协方差部分参数
        {'params': self.__likelihood.parameters(), 'lr': 0.1},         # 仅包含似然部分参数
        ])

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.__likelihood, self.__model)

        # 训练循环
        for i in tqdm(range(self.__iterations), desc="Training GPyTorch GP", unit="iter"):
            optimizer.zero_grad()
            output = self.__model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        # 5. 预测
        self.__model.eval()
        self.__likelihood.eval()

        test_x = torch.from_numpy(x_pred).float()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.__likelihood(self.__model(test_x))
            y_mean = observed_pred.mean.numpy()
            y_cov = observed_pred.variance.numpy()

        # 存储核函数以便查看
        self.__kernels.append(self.__model.covar_module)

        return x_mesh, y_mean, y_cov

    def get_kernels(self):
        """
        返回所使用的核函数列表。
        """
        return self.__kernels