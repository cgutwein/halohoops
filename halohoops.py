"""ncaa basketball prediction model

Tasks to be performed utilizing the class:
- generate training, development, and test data
- train model
- evaluate results
- generate prediction .csv for submittal

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

dirs = {1: {False: './data/MDataFiles_Stage1/M',
            True: './data/WDataFiles_Stage1/W'},
        2: {False: './data/MDataFiles_Stage2/M',
            True: './data/WDataFiles_Stage2/W'}}

class haloHoops:
    """
    prediction model
    """

    def __init__(self, phase=2, w=False):
        self.data = {
            'ordinals': None,
            'seasons': None,
            'reg_results': None,
            'seeds': None,
            'teams': None,
            't_games_compact': None
        }
        self.data_dir = dirs[phase][w]
        self.phase = phase
        self.w = w

    def get_data(self):
        """
        Load data into memory.
        Process data into dataframe such that each row has the following features:
        - KenPom ranking
        - tournament seed
        Women's Tournament data has no rankings so we don't add these.
        """
        self.data['reg_results_c'] = pd.read_csv('{}RegularSeasonCompactResults.csv'.format(self.data_dir))
        # generate win percentage from compact reults
        self.data['reg_results_c'].drop(['NumOT', 'WLoc'], axis=1, inplace=True)
        self.data['reg_results_c']['ScoreGap'] = self.data['reg_results_c']['WScore'] - self.data['reg_results_c']['LScore']
        num_win = self.data['reg_results_c'].groupby(['Season', 'WTeamID']).count()
        num_win = num_win.reset_index()[['Season', 'WTeamID', 'DayNum']].rename(columns={"DayNum": "NumWins", "WTeamID": "TeamID"})
        num_loss = self.data['reg_results_c'].groupby(['Season', 'LTeamID']).count()
        num_loss = num_loss.reset_index()[['Season', 'LTeamID', 'DayNum']].rename(columns={"DayNum": "NumLosses", "LTeamID": "TeamID"})
        gap_win = self.data['reg_results_c'].groupby(['Season', 'WTeamID']).mean().reset_index()
        gap_win = gap_win[['Season', 'WTeamID', 'ScoreGap']].rename(columns={"ScoreGap": "GapWins", "WTeamID": "TeamID"})
        gap_loss = self.data['reg_results_c'].groupby(['Season', 'LTeamID']).mean().reset_index()
        gap_loss = gap_loss[['Season', 'LTeamID', 'ScoreGap']].rename(columns={"ScoreGap": "GapLosses", "LTeamID": "TeamID"})
        df_features_season_w = self.data['reg_results_c'].groupby(['Season', 'WTeamID']).count().reset_index()[['Season', 'WTeamID']].rename(columns={"WTeamID": "TeamID"})
        df_features_season_l = self.data['reg_results_c'].groupby(['Season', 'LTeamID']).count().reset_index()[['Season', 'LTeamID']].rename(columns={"LTeamID": "TeamID"})
        self.df_team_reg = pd.concat([df_features_season_w, df_features_season_l], 0).drop_duplicates().sort_values(['Season', 'TeamID']).reset_index(drop=True)
        self.df_team_reg = self.df_team_reg.merge(num_win, on=['Season', 'TeamID'], how='left')
        self.df_team_reg = self.df_team_reg.merge(num_loss, on=['Season', 'TeamID'], how='left')
        self.df_team_reg = self.df_team_reg.merge(gap_win, on=['Season', 'TeamID'], how='left')
        self.df_team_reg = self.df_team_reg.merge(gap_loss, on=['Season', 'TeamID'], how='left')
        self.df_team_reg.fillna(0, inplace=True)
        # calculate using weighted average
        self.df_team_reg['WinRatio'] = self.df_team_reg['NumWins'] / (self.df_team_reg['NumWins'] + self.df_team_reg['NumLosses'])
        self.df_team_reg['GapAvg'] = (
            (self.df_team_reg['NumWins'] * self.df_team_reg['GapWins'] -
           self.df_team_reg['NumLosses'] * self.df_team_reg['GapLosses'])
            / (self.df_team_reg['NumWins'] + self.df_team_reg['NumLosses'])
        )
        self.df_team_reg.drop(['NumWins', 'NumLosses', 'GapWins', 'GapLosses'], axis=1, inplace=True)

        self.data['reg_results'] = pd.read_csv('{}RegularSeasonDetailedResults.csv'.format(self.data_dir))
        self.data['seasons'] = pd.read_csv('{}Seasons.csv'.format(self.data_dir))
        self.data['seeds'] = pd.read_csv('{}NCAATourneySeeds.csv'.format(self.data_dir))
        self.data['teams'] = pd.read_csv('{}Teams.csv'.format(self.data_dir))
        self.data['t_games_compact'] = pd.read_csv('{}NCAATourneyCompactResults.csv'.format(self.data_dir))
        self.submission = pd.read_csv('{}SampleSubmissionStage{}.csv'.format(self.data_dir, self.phase))
        if not self.w:
            self.data['ordinals'] = pd.read_csv('{}MasseyOrdinals.csv'.format(self.data_dir))
            join_week = dict()
            o_seasons = list(self.data['ordinals']['Season'].unique())
            for s in o_seasons:
                w_max = np.max(self.data['ordinals']['RankingDayNum'][self.data['ordinals']['Season'] == s].unique())
                join_week[s] = w_max
            test_cols = ['ID', 'Season', 't1_id', 't2_id', 't1_kpom', 't2_kpom', 't1_seed', 't2_seed']
        else:
            o_seasons = list(self.data['seeds']['Season'].unique())
            test_cols = ['ID', 'Season', 't1_id', 't2_id', 't1_seed', 't2_seed']
        print("Data Loaded successfully.")

        print("Generating training data frame.")
        df_train = self.data['t_games_compact'].copy()
        df_train = self.data['t_games_compact'].copy()
        df_train = df_train.drop(['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], axis=1)
        #df_train = df_train[(df_train['Season'] > 2002) & (df_train['Season'] < 2016)]
        df_train = df_train[df_train['Season'] > 2002]
        df_train_1 = df_train.copy()
        df_train['Result'] = 1
        df_train = df_train.rename(index=str, columns={"WTeamID": "TeamID_1", "LTeamID": "TeamID_2"})
        df_train_1['Result'] = 0
        df_train_1 = df_train_1.rename(index=str, columns={"WTeamID": "TeamID_2", "LTeamID": "TeamID_1"})
        df_train = pd.concat([df_train, df_train_1], sort=False, ignore_index=True)
        if not self.w:
            df_train['MaxWeek'] = df_train['Season'].apply(lambda x: join_week[int(x)])
            # get KenPom rankings
            df_train['t1_kpom'] = df_train.merge(self.data['ordinals'][self.data['ordinals']['SystemName'] == 'POM'],
                                                  left_on=['Season', 'MaxWeek', 'TeamID_1'],
                                  right_on=['Season', 'RankingDayNum', 'TeamID'],
                                  how='left')['OrdinalRank']
            df_train['t2_kpom'] = df_train.merge(self.data['ordinals'][self.data['ordinals']['SystemName'] == 'POM'],
                                  left_on=['Season', 'MaxWeek', 'TeamID_2'],
                                  right_on=['Season', 'RankingDayNum', 'TeamID'],
                                  how='left')['OrdinalRank']
        df_train['t1_seed'] = df_train.merge(self.data['seeds'],
                                           left_on=['Season', 'TeamID_1'],
                                           right_on=['Season', 'TeamID'],
                                           how='left')['Seed'].astype(str).apply(lambda x: x[1:3]).astype(int)
        df_train['t1_Wavg'] = df_train.merge(self.df_team_reg,
                                           left_on=['Season', 'TeamID_1'],
                                           right_on=['Season', 'TeamID'],
                                           how='left')['WinRatio']
        df_train['t1_margin'] = df_train.merge(self.df_team_reg,
                                           left_on=['Season', 'TeamID_1'],
                                           right_on=['Season', 'TeamID'],
                                           how='left')['GapAvg']
        df_train['t2_seed'] = df_train.merge(self.data['seeds'],
                                           left_on=['Season', 'TeamID_2'],
                                           right_on=['Season', 'TeamID'],
                                           how='left')['Seed'].astype(str).apply(lambda x: x[1:3]).astype(int)
        df_train['t2_Wavg'] = df_train.merge(self.df_team_reg,
                                           left_on=['Season', 'TeamID_2'],
                                           right_on=['Season', 'TeamID'],
                                           how='left')['WinRatio']
        df_train['t2_margin'] = df_train.merge(self.df_team_reg,
                                           left_on=['Season', 'TeamID_2'],
                                           right_on=['Season', 'TeamID'],
                                           how='left')['GapAvg']
        # trim down further if in phase 1 of comp.
        if self.phase == 1:
            df_train = df_train[df_train['Season'] < 2016]
        self.df_train = df_train

        print("Generating test data frame.")
        df_test = pd.DataFrame(columns=test_cols)
        df_test['ID'] = self.submission['ID']
        df_test['Season'] = self.submission['ID'].apply(lambda x: x[0:4]).astype(int)
        df_test['t1_id'] = self.submission['ID'].apply(lambda x: x[5:9]).astype(int)
        df_test['t2_id'] = self.submission['ID'].apply(lambda x: x[10::]).astype(int)
        # join max week to dataframe
        if not self.w:
            df_test['MaxWeek'] = df_test['Season'].apply(lambda x: join_week[int(x)])
            df_test['t1_kpom'] = df_test.merge(self.data['ordinals'][self.data['ordinals']['SystemName'] == 'POM'],
                          left_on=['Season', 'MaxWeek', 't1_id'],
                          right_on=['Season', 'RankingDayNum', 'TeamID'],
                          how='left')['OrdinalRank']
            df_test['t2_kpom'] = df_test.merge(self.data['ordinals'][self.data['ordinals']['SystemName'] == 'POM'],
                          left_on=['Season', 't2_id', 'MaxWeek'],
                          right_on=['Season', 'TeamID', 'RankingDayNum'],
                          how='left', validate='m:1')['OrdinalRank']
        df_test['t1_seed'] = df_test.merge(self.data['seeds'],
                                           left_on=['Season', 't1_id'],
                                           right_on=['Season', 'TeamID'],
                                           how='left')['Seed'].apply(lambda x: x[1:3]).astype(int)
        df_test['t1_Wavg'] = df_test.merge(self.df_team_reg,
                                           left_on=['Season', 't1_id'],
                                           right_on=['Season', 'TeamID'],
                                           how='left')['WinRatio']
        df_test['t1_margin'] = df_test.merge(self.df_team_reg,
                                           left_on=['Season', 't1_id'],
                                           right_on=['Season', 'TeamID'],
                                           how='left')['GapAvg']
        df_test['t2_seed'] = df_test.merge(self.data['seeds'],
                                           left_on=['Season', 't2_id'],
                                           right_on=['Season', 'TeamID'],
                                           how='left')['Seed'].apply(lambda x: x[1:3]).astype(int)
        df_test['t2_Wavg'] = df_test.merge(self.df_team_reg,
                                           left_on=['Season', 't2_id'],
                                           right_on=['Season', 'TeamID'],
                                           how='left')['WinRatio']
        df_test['t2_margin'] = df_test.merge(self.df_team_reg,
                                           left_on=['Season', 't2_id'],
                                           right_on=['Season', 'TeamID'],
                                           how='left')['GapAvg']
        self.test_data = df_test

    def train_model(self):
        """
        Train prediction model.

        Basic logistic regression to start us off.
        """
        if not self.w:
            f_cols = ['Season', 't1_kpom', 't2_kpom', 't1_seed', 't1_Wavg', 't1_margin',
                      't2_seed', 't2_Wavg', 't2_margin']
            d_cols = ['MaxWeek', 'Result', 'TeamID_1', 'TeamID_2']
        else:
            f_cols = ['Season', 't1_seed', 't1_Wavg', 't1_margin',
                      't2_seed', 't2_Wavg', 't2_margin']
            d_cols = ['Result', 'TeamID_1', 'TeamID_2']
        y = self.df_train['Result']
        X = self.df_train.drop(d_cols, axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # get probs for test set
        # calc score for baseline and test set probs
        X_test = self.test_data[f_cols]
        X_test_scaled = scaler.transform(X_test)

        cm = LogisticRegression()
        cm.fit(X_scaled, y)
        self.model = cm
        self.test_preds = cm.predict_proba(X_test_scaled)

        print("Baseline logistic model trained, predictions made...")
        print("Baseline log-loss score of {}".format(0.56336))

    def create_sub(self, filename):
        """
        Generate .csv submission file
        """
        if self.w:
            sub1 = 'W'
        else:
            sub1 = 'M'

        if self.phase == 1:
            sub2 = 'phase1'
        else:
            sub2 = 'phase2'

        self.submission['Pred'] = self.test_preds[:,1]
        self.submission.to_csv('./submissions/{}/{}/{}'.format(sub1, sub2, filename), index=False)
