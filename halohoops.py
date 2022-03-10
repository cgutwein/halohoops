"""ncaa basketball prediction model

Tasks to be performed utilizing the class:
- generate training, development, and test data
- train model
- evaluate results
- generate prediction .csv for submittal

"""
from hh_util import generate_compact, generate_metrics
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
        self.df_team_reg = generate_compact(self.data['reg_results_c'])
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
        self.df_metrics, self.df_feat = generate_metrics(self.data['reg_results'])
        print("Data Loaded successfully.")

        print("Generating training data frame.")
        df_train = self.data['t_games_compact'].copy()
        df_train = self.data['t_games_compact'].copy()
        df_train = df_train.drop(['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], axis=1)
        #df_train = df_train[(df_train['Season'] > 2002) & (df_train['Season'] < 2016)]
        if not self.w:
            df_train = df_train[df_train['Season'] > 2002]
        else:
            df_train = df_train[df_train['Season'] > 2010] # women's regular season data not available prior to 2010
        df_train_1 = df_train.copy()
        df_train['Result'] = 1
        df_train = df_train.rename(index=str, columns={"WTeamID": "TeamID_1", "LTeamID": "TeamID_2"})
        # Join metrics
        df_train = df_train.merge(self.df_metrics, left_on=['Season', 'TeamID_1'], right_on=['Season', 'WTeamID'], how='left')
        df_train = df_train.merge(self.df_metrics, left_on=['Season', 'TeamID_2'], right_on=['Season', 'WTeamID'], how='left', suffixes=[None, 'B'])
        df_train_1['Result'] = 0
        df_train_1 = df_train_1.rename(index=str, columns={"WTeamID": "TeamID_2", "LTeamID": "TeamID_1"})
        df_train_1 = df_train_1.merge(self.df_metrics, left_on=['Season', 'TeamID_2'], right_on=['Season', 'WTeamID'], how='left')
        df_train_1 = df_train_1.merge(self.df_metrics, left_on=['Season', 'TeamID_1'], right_on=['Season', 'WTeamID'], how='left', suffixes=[None, 'B'])
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
            df_train['dkpom'] = df_train['t1_kpom'] - df_train['t2_kpom']
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

        # Calculate margins
        df_train['dPossessions'] = df_train['Possessions'] - df_train['PossessionsB']
        df_train['dPtsPerPoss'] = df_train['PtsPerPoss'] - df_train['PtsPerPossB']
        df_train['dEffectiveFGPct'] = df_train['EffectiveFGPct'] - df_train['EffectiveFGPctB']
        df_train['dAssistRate'] = df_train['AssistRate'] - df_train['AssistRateB']
        df_train['dOReboundPct'] = df_train['OReboundPct'] - df_train['OReboundPctB']
        df_train['dDReboundPct'] = df_train['DReboundPct'] - df_train['DReboundPctB']
        df_train['dATORatio'] = df_train['ATORatio'] - df_train['ATORatioB']
        df_train['dTORate'] = df_train['TORate'] - df_train['TORateB']
        df_train['dBArcPct'] = df_train['BArcPct'] - df_train['BArcPctB']
        df_train['dFTRate'] = df_train['FTRate'] - df_train['FTRateB']
        df_train['dBlockFoul'] = df_train['BlockFoul'] - df_train['BlockFoulB']
        df_train['dStealFoul'] = df_train['StealFoul'] - df_train['StealFoulB']
        df_train['dseed'] = df_train['t1_seed'] - df_train['t2_seed']
        df_train['dWavg'] = df_train['t1_Wavg'] - df_train['t2_Wavg']
        df_train['dmargin'] = df_train['t1_margin'] - df_train['t2_margin']

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
        # merge metrics for each team ID
        df_test = df_test.merge(self.df_metrics, left_on=['Season', 't1_id'], right_on=['Season', 'WTeamID'], how='left')
        df_test = df_test.merge(self.df_metrics, left_on=['Season', 't2_id'], right_on=['Season', 'WTeamID'], how='left', suffixes=[None, 'B'])
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
            df_test['dkpom'] = df_test['t1_kpom'] - df_test['t2_kpom']
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

        # Calculate margins
        df_test['dPossessions'] = df_test['Possessions'] - df_test['PossessionsB']
        df_test['dPtsPerPoss'] = df_test['PtsPerPoss'] - df_test['PtsPerPossB']
        df_test['dEffectiveFGPct'] = df_test['EffectiveFGPct'] - df_test['EffectiveFGPctB']
        df_test['dAssistRate'] = df_test['AssistRate'] - df_test['AssistRateB']
        df_test['dOReboundPct'] = df_test['OReboundPct'] - df_test['OReboundPctB']
        df_test['dDReboundPct'] = df_test['DReboundPct'] - df_test['DReboundPctB']
        df_test['dATORatio'] = df_test['ATORatio'] - df_test['ATORatioB']
        df_test['dTORate'] = df_test['TORate'] - df_test['TORateB']
        df_test['dBArcPct'] = df_test['BArcPct'] - df_test['BArcPctB']
        df_test['dFTRate'] = df_test['FTRate'] - df_test['FTRateB']
        df_test['dBlockFoul'] = df_test['BlockFoul'] - df_test['BlockFoulB']
        df_test['dStealFoul'] = df_test['StealFoul'] - df_test['StealFoulB']
        df_test['dseed'] = df_test['t1_seed'] - df_test['t2_seed']
        df_test['dWavg'] = df_test['t1_Wavg'] - df_test['t2_Wavg']
        df_test['dmargin'] = df_test['t1_margin'] - df_test['t2_margin']

        self.test_data = df_test

    def train_model(self):
        """
        Train prediction model.

        Basic logistic regression to start us off.
        """
        if not self.w:
            f_cols = ['dPossessions', 'dPtsPerPoss', 'dEffectiveFGPct', 'dAssistRate',
                      'dOReboundPct', 'dDReboundPct', 'dATORatio', 'dTORate', 'dBArcPct',
                      'dFTRate', 'dBlockFoul', 'dStealFoul', 'dseed', 'dWavg', 'dmargin',
                      'dkpom']
            d_cols = ['MaxWeek', 'Result', 'TeamID_1', 'TeamID_2']
        else:
            f_cols = ['dPossessions', 'dPtsPerPoss', 'dEffectiveFGPct', 'dAssistRate',
                      'dOReboundPct', 'dDReboundPct', 'dATORatio', 'dTORate', 'dBArcPct',
                      'dFTRate', 'dBlockFoul', 'dStealFoul', 'dseed', 'dWavg', 'dmargin']
            d_cols = ['Result', 'TeamID_1', 'TeamID_2']
        y = self.df_train['Result']
        X = self.df_train[f_cols]
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
