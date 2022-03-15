"""
Utility functions for halo hoops.
"""

from tkinter.tix import DirSelectBox
from numpy import result_type
import pandas as pd

def weighted_metric(metric, df):
    """Computes the weighted stat from winning and losing metric"""

    weighted_df = ((df[f'W{metric}'].mul(df['NumWins']) + df[f'L{metric}'].mul(df['NumLosses'])) / (df['NumWins'] + df['NumLosses']))
    return(weighted_df)

def generate_compact(df_com):
    """
    Generate win percentage and scoring margins.

    parameters
    ----------
    df_com: (pandas dataframe)
        regular season compact results
    """
    df_com.drop(['NumOT', 'WLoc'], axis=1, inplace=True)
    df_com['ScoreGap'] = df_com['WScore'] - df_com['LScore']
    num_win = df_com.groupby(['Season', 'WTeamID']).count()
    num_win = num_win.reset_index()[['Season', 'WTeamID', 'DayNum']].rename(columns={"DayNum": "NumWins", "WTeamID": "TeamID"})
    num_loss = df_com.groupby(['Season', 'LTeamID']).count()
    num_loss = num_loss.reset_index()[['Season', 'LTeamID', 'DayNum']].rename(columns={"DayNum": "NumLosses", "LTeamID": "TeamID"})
    gap_win = df_com.groupby(['Season', 'WTeamID']).mean().reset_index()
    gap_win = gap_win[['Season', 'WTeamID', 'ScoreGap']].rename(columns={"ScoreGap": "GapWins", "WTeamID": "TeamID"})
    gap_loss = df_com.groupby(['Season', 'LTeamID']).mean().reset_index()
    gap_loss = gap_loss[['Season', 'LTeamID', 'ScoreGap']].rename(columns={"ScoreGap": "GapLosses", "LTeamID": "TeamID"})
    df_features_season_w = df_com.groupby(['Season', 'WTeamID']).count().reset_index()[['Season', 'WTeamID']].rename(columns={"WTeamID": "TeamID"})
    df_features_season_l = df_com.groupby(['Season', 'LTeamID']).count().reset_index()[['Season', 'LTeamID']].rename(columns={"LTeamID": "TeamID"})
    df_out = pd.concat([df_features_season_w, df_features_season_l], 0).drop_duplicates().sort_values(['Season', 'TeamID']).reset_index(drop=True)
    df_out = df_out.merge(num_win, on=['Season', 'TeamID'], how='left')
    df_out = df_out.merge(num_loss, on=['Season', 'TeamID'], how='left')
    df_out = df_out.merge(gap_win, on=['Season', 'TeamID'], how='left')
    df_out = df_out.merge(gap_loss, on=['Season', 'TeamID'], how='left')
    df_out.fillna(0, inplace=True)
    # calculate using weighted average
    df_out['WinRatio'] = df_out['NumWins'] / (df_out['NumWins'] + df_out['NumLosses'])
    df_out['GapAvg'] = (
        (df_out['NumWins'] * df_out['GapWins'] -
       df_out['NumLosses'] * df_out['GapLosses'])
        / (df_out['NumWins'] + df_out['NumLosses'])
    )
    df_out.drop(['NumWins', 'NumLosses', 'GapWins', 'GapLosses'], axis=1, inplace=True)
    return(df_out)

def generate_metrics(df_reg):
    """
    Generate regular season statistics from detailed results file.

    parameters
    ----------
    df_reg: (pandas dataframe)
        regular season detailed (i.e. box score) results
    """
    # adding sabermetrics per kaggle notebook
    df_reg.drop(['NumOT', 'WLoc'], axis=1, inplace=True)
    df_reg['ScoreMargin'] = df_reg['WScore'] - df_reg['LScore']
    factor, vop, drbp, pf, fta, ftm = calculate_efficiency_constants(df_reg)

    num_win = df_reg.groupby(['Season', 'WTeamID']).count()
    num_win = num_win.reset_index()[['Season', 'WTeamID', 'DayNum']].rename(columns={"DayNum": "NumWins", "WTeamID": "TeamID"}).fillna(0)
    win_score_margin = df_reg.groupby(['Season', 'WTeamID']).mean().reset_index()
    win_score_margin = win_score_margin[['Season', 'WTeamID', 'ScoreMargin']].rename(columns={"ScoreMargin": "AvgWinningScoreMargin", "WTeamID": "TeamID"}).fillna(0)

    num_loss = df_reg.groupby(['Season', 'LTeamID']).count()
    num_loss = num_loss.reset_index()[['Season', 'LTeamID', 'DayNum']].rename(columns={"DayNum": "NumLosses", "LTeamID": "TeamID"}).fillna(0)
    lose_score_margin = df_reg.groupby(['Season', 'LTeamID']).mean().reset_index()
    lose_score_margin = lose_score_margin[['Season', 'LTeamID', 'ScoreMargin']].rename(columns={"ScoreMargin": "AvgLosingScoreMargin", "LTeamID": "TeamID"}).fillna(0)

    df_features_season_w = df_reg.groupby(['Season', 'WTeamID']).count().reset_index()[['Season', 'WTeamID']].rename(columns={"WTeamID": "TeamID"})
    df_features_season_l = df_reg.groupby(['Season', 'LTeamID']).count().reset_index()[['Season', 'LTeamID']].rename(columns={"LTeamID": "TeamID"})

    df_features_season = pd.concat([df_features_season_w, df_features_season_l], axis=0).drop_duplicates().sort_values(['Season', 'TeamID']).reset_index(drop=True)
    df_features_season = df_features_season.merge(num_win, on=['Season', 'TeamID'], how='left')
    df_features_season = df_features_season.merge(num_loss, on=['Season', 'TeamID'], how='left')
    df_features_season = df_features_season.merge(win_score_margin, on=['Season', 'TeamID'], how='left')
    df_features_season = df_features_season.merge(lose_score_margin, on=['Season', 'TeamID'], how='left')

    # fill in missing values
    df_features_season['NumWins'] = df_features_season['NumWins'].fillna(0)
    df_features_season['NumLosses'] = df_features_season['NumLosses'].fillna(0)
    df_features_season['AvgWinningScoreMargin'] = df_features_season['AvgWinningScoreMargin'].fillna(0)
    df_features_season['AvgLosingScoreMargin'] = df_features_season['AvgLosingScoreMargin'].fillna(0)

    # Calc winning percentage and margin
    df_features_season['WinPercentage'] = df_features_season['NumWins'] / (df_features_season['NumWins'] + df_features_season['NumLosses'])
    df_features_season['AvgScoringMargin'] = (
        (df_features_season['NumWins'] * df_features_season['AvgWinningScoreMargin'] -
        df_features_season['NumLosses'] * df_features_season['AvgLosingScoreMargin'])
        / (df_features_season['NumWins'] + df_features_season['NumLosses'])
    )

    df_features_season.drop(['AvgWinningScoreMargin', 'AvgLosingScoreMargin'], axis=1, inplace=True)

    sabermetrics = pd.DataFrame()

    sabermetrics['Season'] = df_reg['Season']
    sabermetrics['WTeamID'] = df_reg['WTeamID']
    sabermetrics['LTeamID'] = df_reg['LTeamID']

    # Number of Possessions
    sabermetrics['WPossessions'] = (df_reg['WFGA'] - df_reg['WOR']) + df_reg['WTO'] + .44 * df_reg['WFTA']
    sabermetrics['LPossessions'] = (df_reg['LFGA'] - df_reg['LOR']) + df_reg['LTO'] + .44 * df_reg['LFTA']

    df_reg['WPossessions'] = sabermetrics['WPossessions']
    df_reg['LPossessions'] = sabermetrics['LPossessions']

    # Points Per Possession
    sabermetrics['WPtsPerPoss'] = df_reg['WScore'] / df_reg['WPossessions']
    sabermetrics['LPtsPerPoss'] = df_reg['LScore'] / df_reg['LPossessions']
    
    # True Shooting Percentage
    sabermetrics['WTShootingPct'] = df_reg['WScore'] / (2 * (df_reg['WFGA'] + .475 * df_reg['WFTA']))
    sabermetrics['LTShootingPct'] = df_reg['LScore'] / (2 * (df_reg['LFGA'] + .475 * df_reg['LFTA']))
    
    # Effective Field Goal Percentage
    sabermetrics['WEffectiveFGPct'] = ((df_reg['WScore'] - df_reg['WFTM']) / 2) / df_reg['WFGA']
    sabermetrics['LEffectiveFGPct'] = ((df_reg['LScore'] - df_reg['LFTM']) / 2) / df_reg['LFGA']

    # Percentage of Field Goals Assisted
    sabermetrics['WAssistRate'] = df_reg['WAst'] / df_reg['WFGM']
    sabermetrics['LAssistRate'] = df_reg['LAst'] / df_reg['LFGM']

    # Offensive Rebound Percentage
    sabermetrics['WOReboundPct'] = df_reg['WOR'] / (df_reg['WFGA'] - df_reg['WFGM'])
    sabermetrics['LOReboundPct'] = df_reg['LOR'] / (df_reg['LFGA'] - df_reg['LFGM'])

    # Defensive Rebound Percentage
    sabermetrics['WDReboundPct'] = df_reg['WDR'] / (df_reg['LFGA'] - df_reg['LFGM'])
    sabermetrics['LDReboundPct'] = df_reg['LDR'] / (df_reg['WFGA'] - df_reg['WFGM'])
    # Assist to Turnover Ratio
    sabermetrics['WATORatio'] = df_reg['WAst'] / df_reg['WTO']
    sabermetrics['LATORatio'] = df_reg['LAst'] / df_reg['LTO']

    # Turnover Rate
    sabermetrics['WTORate'] = df_reg['WTO'] / df_reg['WPossessions']
    sabermetrics['LTORate'] = df_reg['LTO'] /  df_reg['LPossessions']

    # Percentage of Shots Beyond the Arc
    sabermetrics['WBArcPct'] = df_reg['WFGA3'] / df_reg['WFGA']
    sabermetrics['LBArcPct'] = df_reg['LFGA3'] /  df_reg['LFGA']

    # Free Throw Rate
    sabermetrics['WFTRate'] = df_reg['WFTA'] / df_reg['WFGA']
    sabermetrics['LFTRate'] = df_reg['LFTA'] /  df_reg['LFGA']

    # Block to Foul Percentage
    sabermetrics['WBlockFoul'] = df_reg['WBlk'] / (df_reg['WPF'] + df_reg['WBlk'])
    sabermetrics['LBlockFoul'] = df_reg['LBlk'] / (df_reg['LPF'] + df_reg['LBlk'])

    # Steal to Foul Percentage
    sabermetrics['WStealFoul'] = df_reg['WStl'] / (df_reg['WPF'] + df_reg['WStl'])
    sabermetrics['LStealFoul'] = df_reg['LStl'] / (df_reg['LPF'] + df_reg['LStl'])
    
    # Team total PER (Player Efficiency)
    #sabermetrics['WTeamPER'] = df_reg.apply(lambda row: row['WFGM3'] - ((row['WPF'] * ftm[ftm['Season'] == row['Season']][0])/pf[pf['Season'] == row['Season']][0]) + (row['WFTA'] / 2 * (2 - (row['WAst'] / 3 * row['WFGM']))) + (row['WFGM'] * (2 - (factor[factor['Season'] == row['Season']][0] * row['WAst']) / row['WFGM'])) + (2 * row['WAst'] / 3) + vop[vop['Season'] == row['Season']][0] * (drbp[drbp['Season'] == row['Season']][0] * (2 * row['WOR'] + row['WBlk'] - 0.2464 * (row['WFTA'] - row['WFTM']) - (row['WFGA'] - row['WFGM']) - (row['WOR'] + row['WDR'])) + ((0.44 * fta[fta['Season'] == row['Season']][0] * row['WPF']) / pf[pf['Season'] == row['Season']][0]) - (row['WTO'] + row['WOR']) + row['WStl'] + row['WOR'] + row['WDR'] - (0.1936 * (row['WFTA'] - row['WFTM']))), axis=1, result_type='reduce')
    #sabermetrics['LTeamPER'] = df_reg['LFGM3'] - ((df_reg['LPF'] * ftm)/pf) + (df_reg['LFTA'] / 2 * (2 - (df_reg['LAst'] / 3 * df_reg['LFGM']))) + (df_reg['LFGM'] * (2 - (factor * df_reg['LAst']) / df_reg['LFGM'])) + (2 * df_reg['LAst'] / 3) + vop * (drbp * (2 * df_reg['LOR'] + df_reg['LBlk'] - 0.2464 * (df_reg['LFTA'] - df_reg['LFTM']) - (df_reg['LFGA'] - df_reg['LFGM']) - (df_reg['LOR'] + df_reg['LDR'])) + ((0.44 * fta * df_reg['LPF']) / pf) - (df_reg['LTO'] + df_reg['LOR']) + df_reg['LStl'] + df_reg['LOR'] + df_reg['LDR'] - (0.1936 * (df_reg['LFTA'] - df_reg['LFTM'])))

    winning_columns = sabermetrics[[col for col in sabermetrics.columns if col[0] == 'W']]
    losing_columns = sabermetrics[[col for col in sabermetrics.columns if col[0] == 'L']]
    winning_columns.loc[:, 'Season'] = sabermetrics['Season']
    losing_columns.loc[:, 'Season'] = sabermetrics['Season']

    winning_sabermetrics = winning_columns.groupby(['Season', 'WTeamID']).mean()
    losing_sabermetrics = losing_columns.groupby(['Season', 'LTeamID']).mean()

    winning_sabermetrics = winning_sabermetrics \
                        .reset_index() \
                        .merge(df_features_season[['Season', 'TeamID', 'NumWins']], left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left') \
                        .set_index(['Season', 'WTeamID']) \
                        .drop(['TeamID'], axis=1)

    losing_sabermetrics = losing_sabermetrics \
                            .reset_index() \
                            .merge(df_features_season[['Season', 'TeamID', 'NumLosses']], left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left') \
                            .set_index(['Season', 'LTeamID']) \
                            .drop(['TeamID'], axis=1)

    weighted_sabermetrics_wins = winning_sabermetrics[[col for col in winning_sabermetrics.columns if col[0] == 'W']].multiply(winning_sabermetrics['NumWins'], axis=0)
    weighted_sabermetrics_losses = losing_sabermetrics[[col for col in losing_sabermetrics.columns if col[0] == 'L']].multiply(losing_sabermetrics['NumLosses'], axis=0)

    weighted_sabermetrics = pd.DataFrame()
    weighted_sabermetrics['Possessions'] = (weighted_sabermetrics_wins['WPossessions'] + weighted_sabermetrics_losses['LPossessions']) /  \
                                           (winning_sabermetrics['NumWins'] + losing_sabermetrics['NumLosses'])

    combined_df = winning_sabermetrics.reset_index().merge(losing_sabermetrics.reset_index(), left_on=['WTeamID', 'Season'], right_on=['LTeamID', 'Season'], how='outer')
    combined_df.reset_index(inplace=True)
    combined_df['WTeamID'].fillna(combined_df['LTeamID'], inplace=True)
    combined_df['LTeamID'].fillna(combined_df['WTeamID'], inplace=True)
    combined_df.set_index(['Season', 'WTeamID'], inplace=True)
    combined_df.fillna(0, inplace=True)
    #combined_df.set_index(['Season', 'WTeamID'], inplace=True)

    metrics_list = ['Possessions', 'PtsPerPoss', 'TrueShootingPct', 'EffectiveFGPct', 'AssistRate', 'OReboundPct', 'DReboundPct', 'ATORatio', 'TORate', 'BArcPct', 'FTRate', 'BlockFoul', 'StealFoul', 'TeamPER']
    season_sabermetrics = pd.concat([weighted_metric(metric, combined_df) for metric in metrics_list], axis=1)
    season_sabermetrics.columns=metrics_list
    season_sabermetrics.sort_index(inplace=True)
    season_sabermetrics.index.columns = ['Season', 'TeamID']
    season_sabermetrics.reset_index(inplace=True)

    return(season_sabermetrics, df_features_season)


def calculate_efficiency_constants(df_reg):
    yearly_df_reg = df_reg.groupby(['Season']).sum()
    
    # Scale Factor
    factor = (2/3) - ((0.5 * ((yearly_df_reg['WAst'] + yearly_df_reg['LAst'])/(yearly_df_reg['WFGM'] + yearly_df_reg['LFGM'])))/(2 * ((yearly_df_reg['WFGM'] + yearly_df_reg['LFGM'])/(yearly_df_reg['WFTM'] + yearly_df_reg['LFTM']))))
   
    # Value of Possession
    vop = (yearly_df_reg['WScore'] + yearly_df_reg['LScore']) / ((yearly_df_reg['WFGA'] + yearly_df_reg['LFGA']) - (yearly_df_reg['WOR'] + yearly_df_reg['LOR']) + (yearly_df_reg['WTO'] + yearly_df_reg['LTO']) + 0.44 * (yearly_df_reg['WFTA'] + yearly_df_reg['LFTA']))
    
    # Def Rebound %
    drbp = (yearly_df_reg['WDR'] + yearly_df_reg['LDR']) / ((yearly_df_reg['WOR'] + yearly_df_reg['LOR']) + (yearly_df_reg['WDR'] + yearly_df_reg['LDR']))
    
    # League total fouls
    pf = yearly_df_reg['WPF'] + yearly_df_reg['LPF']
    
    # League Free Throw Attempts
    fta = yearly_df_reg['WFTA'] + yearly_df_reg['LFTA']
    
    # League Free Throws Made
    ftm = yearly_df_reg['WFTM'] + yearly_df_reg['LFTM']
    
    factor = factor.reset_index()
    vop = vop.reset_index()
    drbp = drbp.reset_index()
    pf = pf.reset_index()
    fta = fta.reset_index()
    ftm = ftm.reset_index()
    return factor, vop, drbp, pf, fta, ftm
