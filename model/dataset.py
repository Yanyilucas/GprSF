import pandas as pd
import numpy as np

class csv_handler:
    df = None
    quarters = None
    years = None
    max_days = None

    def __init__(self, csv_name: str):
        if csv_name == 'COMP':
            __import__('ipdb').set_trace()
        self.__load_data(csv_name)
        self.df['Norm Adj Close'] = self.__add_normalized_data(self.df)
        self.df['Quarter'] = self.__add_quarters(self.df)
        self.max_days = 256

    def get_equal_length_prices(self, normalized=True):
        df = self.__shift_first_year_prices()
        for i in range(1, len(self.years)):
            df = pd.concat([df, pd.DataFrame(self.get_year_data(year=self.years[i], normalized=normalized))], axis=1)

        df = df[:self.max_days]

        quarters = []
        for j in range(0, len(self.quarters)):
            for i in range(0, self.max_days // 4):
                quarters.append(self.quarters[j])
        quarters = pd.DataFrame(quarters)

        df = pd.concat([df, quarters], axis=1)
        df.columns = self.years + ['Quarter']
        df.index.name = 'Day'

        self.__fill_last_rows(df)

        return df

    def get_year_data(self, year: int, normalized=True):
        if year not in self.years:
            raise ValueError('\n' +
                             'Input year: {} not in available years: {}'.format(year, self.years))

        prices = (self.df.loc[self.df['Date'].dt.year == year])
        if normalized:
            return np.asarray(prices.loc[:, 'Norm Adj Close'])
        else:
            return np.asarray(prices.loc[:, 'Adj Close'])

    def get_whole_prices(self, start_year: int, end_year: int):
        if start_year < self.years[0] or end_year > self.years[-1]:
            raise ValueError('\n' +
                             'Input years out of available range! \n' +
                             'Max range available: {}-{}\n'.format(self.years[0], self.years[-1]) +
                             'Was: {}-{}'.format(start_year, end_year))

        df = (self.df.loc[(self.df['Date'].dt.year >= start_year) & (self.df['Date'].dt.year <= end_year)])
        df = df.loc[:, ['Date', 'Adj Close']]

        return df

    def show(self, max_rows=None, max_columns=None):
        with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_columns):
            print(self.df)

    # def __load_data(self, csv_name: str):
    #     self.df = pd.read_csv('Data/' + csv_name + '.csv')
    #     self.df = self.df.iloc[:, [0, 5]]
    #     self.df = self.df.dropna()
    #     self.df.Date = pd.to_datetime(self.df.Date)
    #     self.quarters = ['Q' + str(i) for i in range(1, 5)]
    def __load_data(self, csv_name: str):
        """
        自动兼容两种 CSV 格式：
        - 旧格式：含有 'Adj Close' 列
        - 新格式：没有 'Adj Close' 列，但含有 'Close/Last' 列
        
        最终将收盘价统一命名为 'Adj Close'，并只保留 'Date' 与 'Adj Close' 两列。
        """
        
        # 1. 读取 CSV
        self.df = pd.read_csv('Data/' + csv_name + '.csv')
        
        # 2. 如果原 CSV 中不存在 'Adj Close' 列，但存在 'Close/Last' 列，则重命名
        if 'Adj Close' not in self.df.columns:
            if 'Close/Last' in self.df.columns:
                self.df.rename(columns={'Close/Last': 'Adj Close'}, inplace=True)
            else:
                raise ValueError(
                    f'CSV 文件没有 "Adj Close" 或 "Close/Last" 列，无法解析: {csv_name}'
                )
        
        # 3. 保留需要用的列：只留 'Date' 和 'Adj Close'
        #   如果后续需要 'Open' / 'High' / 'Low' 等列，可根据需求保留
        self.df = self.df[['Date', 'Adj Close']]

        # 4. 清理缺失值
        self.df.dropna(inplace=True)

        # 5. 转换日期类型（自动解析；若有格式异常可指定 format='%m/%d/%Y' ）
        self.df['Date'] = pd.to_datetime(self.df['Date'])

        # 6. 生成四个季度标签（Q1 ~ Q4），供后续逻辑使用
        self.quarters = ['Q' + str(i) for i in range(1, 5)]

    def __add_normalized_data(self, df):
        normalized_list = []

        self.years = list(df.Date)
        self.years = list({year.year for year in self.years})

        for year in self.years:
            prices = self.get_year_data(year=year, normalized=False)
            mean = np.mean(prices)
            std = np.std(prices)
            prices = [(p - mean) / std for p in prices]
            prices = [p - prices[0] for p in prices]
            normalized_list.append(prices)

        # 将所有价格列表转换为 DataFrame 并合并
        normalized = pd.concat([pd.DataFrame(lst) for lst in normalized_list], ignore_index=True)

        return normalized
    def __add_quarters(self, df):
        quarters_list = []

        for year in self.years:
            # 筛选特定年份的数据
            dates = df.loc[df['Date'].dt.year == year, 'Date']
            # 获取每个日期对应的季度
            quarters = [self.__get_quarter(date.month) for date in dates]
            # 将季度数据转换为 DataFrame 并添加到列表中
            quarters_df = pd.DataFrame(quarters)
            quarters_list.append(quarters_df)

        # 一次性合并所有季度 DataFrame
        quarters = pd.concat(quarters_list, ignore_index=True)

        return quarters

    def __get_quarter(self, month: int):
        return self.quarters[(month - 1) // 3]

    def __shift_first_year_prices(self):
        prices = pd.DataFrame(self.get_year_data(self.years[0]))
        df = pd.DataFrame([0 for _ in range(self.max_days - len(prices.index))])
        df = pd.concat([df, prices], ignore_index=True)

        return df

    def __fill_last_rows(self, df):
        years = self.years[:-1]

        for year in years:
            mean = np.mean(df[year])
            for i in range(self.max_days - 1, -1, -1):
                current_price = df.iloc[i, df.columns.get_loc(year)]
                if np.isnan(current_price):
                    df.iloc[i, df.columns.get_loc(year)] = mean
                else:
                    break
