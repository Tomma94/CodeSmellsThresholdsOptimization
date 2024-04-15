import pandas as pd

# select the relevance column from the all data
selected_columns = ['id', 'type_metrics_NOF', 'type_metrics_NOM',
                    'type_metrics_NOPM',
                    'type_metrics_DIT', 'type_metrics_LCOM',
                    'type_metrics_FANIN', 'type_metrics_FANOUT',
                    'organic_type_metrics_CLOC', 'organic_type_metrics_IsAbstract',
                    'organic_type_metrics_OverrideRatio',
                    'organic_type_metrics_PublicFieldCount', 'organic_type_metrics_TCC',
                    'organic_type_metrics_NOAM', 'organic_type_metrics_WMC',
                    'organic_type_metrics_WOC',
                    'organic_type_metrics_InterfaceMethodDeclarationCount',
                    'organic_type_metrics_IsSingleton',
                    'organic_type_metrics_AccessorCount', 'is_buggy', 'Project',
                    'Version']

new_columns_name = ['id', 'NumberOfFields', 'NumberOfMethods',
                              'NumberOfPublicMethods',
                              'DepthOfInheritance', 'LCOM',
                              'FANIN', 'FANOUT',
                              'LOCClass', 'IsAbstract',
                              'OverrideRatio',
                              'PublicFieldCount', 'TCC',
                              'NOAM', 'WMC',
                              'WOC',
                              'InterfaceMethodDeclarationCount',
                              'IsSingleton',
                              'AccessorCount', 'Bugged', 'Project',
                              'Version']

# smells and metrics lists- can be modified base on the smells you want and the data you have
numeric_metrics_list = ['InterfaceMethodDeclarationCount', 'NOAM', 'WOC', 'WMC', 'OverrideRatio', 'PublicFieldCount',
           'AccessorCount', 'TCC', 'LOCClass',
           'LCOM', 'NumberOfFields', 'NumberOfPublicMethods', 'FANIN', 'FANOUT', 'DepthOfInheritance',
           'NumberOfMethods']

metrics_to_normalize = ['LOCClass_zscore', 'OverrideRatio',
                        'TCC_zscore', 'LCOM', 'NumberOfFields_zscore', 'NumberOfMethods_zscore',
                        'FANIN_zscore', 'FANOUT_zscore', 'DepthOfInheritance_zscore',
                        'InterfaceMethodDeclarationCount_zscore', 'PublicFieldCount_zscore']

# A dictionary of the smells- {code smells: [the relevance metrics]}
smells_dict = {'lazy_class': ['LOCClass_zscore'],
               'refused_bequest': ['OverrideRatio'],
               'large_class': ['LOCClass_zscore'],
               'god_class': ['LOCClass_zscore', 'TCC_zscore'],
               'multifaceted_abstraction': ['LCOM', 'NumberOfFields_zscore', 'NumberOfMethods_zscore'],
               'hub-like_modularization': ['FANIN_zscore', 'FANOUT_zscore'],
               'deep_hierarchy': ['DepthOfInheritance_zscore'],
               'swiss_army_knife': ['IsAbstract', 'InterfaceMethodDeclarationCount_zscore'],
               'unnecessary_abstraction': ['NumberOfPublicMethods_bool', 'NumberOfFields_zscore'],
               'broken_modularization': ['NumberOfPublicMethods_bool', 'NumberOfFields_zscore'],
               'class_data_should_be_private': ['PublicFieldCount_zscore'],
               }


smells_list = ['lazy_class', 'refused_bequest', 'large_class', 'god_class', 'multifaceted_abstraction',
               'hub-like_modularization', 'deep_hierarchy', 'swiss_army_knife', 'unnecessary_abstraction',
               'broken_modularization', 'class_data_should_be_private']


def calc_code_smells(data, smells_list):
    '''
    This function calculate the code smells base on the origin thresholds.
    It is modified the dataframe 'data'- add the code smells columns.
    It is necessary to add manually more code smells if it needed.
    :param data: the dataframe with the relevance metrics for this code smells
    :param smells_list: a list of the code smells that this function calc.
    '''
    data['lazy_class'] = data['LOCClass'] < data['LOCClass_perc25']
    data['refused_bequest'] = data['OverrideRatio'] > 0.5
    data['large_class'] = data['LOCClass'] > 1.5 * (data['LOCClass_avg'] + data['LOCClass_std'])
    data['god_class'] = (data['LOCClass'] > 500) & (data['TCC'] < data['TCC_avg'])
    data['multifaceted_abstraction'] = (
                (data['LCOM'] > 0.8) & (data['NumberOfFields'] > 7) & (data['NumberOfMethods'] > 7))
    data['hub-like_modularization'] = ((data['FANIN'] > 20) & (data['FANOUT'] > 20))
    data['deep_hierarchy'] = data['DepthOfInheritance'] > 6
    data['swiss_army_knife'] = ((data['IsAbstract'] == 1) & (data['InterfaceMethodDeclarationCount'] > 1.5 * (
                data['InterfaceMethodDeclarationCount_avg'] + data['InterfaceMethodDeclarationCount_std'])))
    data['unnecessary_abstraction'] = ((data['NumberOfMethods'] == 0) & (data['NumberOfFields'] > 5))
    data['broken_modularization'] = ((data['NumberOfMethods'] == 0) & (data['NumberOfFields'] < 5))
    data['class_data_should_be_private'] = data['PublicFieldCount'] > 0

    for column in smells_list:
        data.replace({column: {True: 1, False: 0}}, inplace=True)

def first_split(df, project_versions_dict):
    """
    4 versions of each project will add to the train, one version to the test
    """

    # Create an empty list to store the rows for each DataFrame
    df_test_rows = []
    df_train_rows = []


    # Iterate over the dictionary items
    for project, versions in project_versions_dict.items():
        # Rows for the first 4 versions go to df_train
        df_train_rows.extend(
            df[(df['Project'] == project) & (df['Version'].isin(versions[:4]))].to_dict('records'))

        # Rows for the fifth version go to df_test
        df_test_rows.extend(df[(df['Project'] == project) & (df['Version'] == versions[4])].to_dict('records'))

    # Create DataFrames from the collected rows
    df_train = pd.DataFrame(df_train_rows)
    df_test = pd.DataFrame(df_test_rows)

    return df_train, df_test

def train_test_projects_version_split(all_data_path, to_save= False):
    # get the data in Data Frame
    df_metrics_old = pd.read_csv(all_data_path)
    df_metrics_old = df_metrics_old.dropna()

    # Specify the columns you want to include in the new DataFrame
    df_metrics_old = df_metrics_old[selected_columns].copy()
    df_metrics_old.columns = new_columns_name

    # Calculate the statistics for numerics metrics
    avgs = df_metrics_old.groupby(['Project', 'Version'])[numeric_metrics_list].mean()
    std = df_metrics_old.groupby(['Project', 'Version'])[numeric_metrics_list].std()
    perc25 = df_metrics_old.groupby(['Project', 'Version'])[numeric_metrics_list].quantile(.25)

    df_metrics = df_metrics_old.merge(avgs, on=['Project', 'Version'], suffixes=['', '_avg']).merge(std, on=['Project',
                                                                                                             'Version'],
                                                                                                    suffixes=['', '_std']).merge(perc25, on=['Project',
                                                                                                             'Version'],
                                                                                         suffixes=['', '_perc25'])
    # Calculate Z-Score for relevance metrics
    for column in numeric_metrics_list:
        # create new column - which represent the Zscore of the metric
        df_metrics[f'{column}_zscore'] = (df_metrics[column] - df_metrics[f'{column}_avg']) / df_metrics[
            f'{column}_std']

    # Deal with booleans metrics
    df_metrics['NumberOfPublicMethods_bool'] = df_metrics['NumberOfPublicMethods'].apply(lambda x: 1 if x == 0 else 0)
    df_metrics['IsAbstract'] = df_metrics['IsAbstract'].apply(lambda x: 0 if x == 0 else 1)

    # Create new DF with only relevance columns (after the calculation)
    smells_list = list(smells_dict.keys())

    # Calculate the origin code smells for the baseline comparison
    calc_code_smells(df_metrics, smells_list)

    # create new df for the relevant metrics:
    df_metrics_new = pd.DataFrame()
    # add the bugged column
    df_metrics_new['Project'] = df_metrics['Project']
    df_metrics_new['Version'] = df_metrics['Version']
    df_metrics_new['id'] = df_metrics['id']
    df_metrics_new['Bugged'] = df_metrics['Bugged']

    # add all relevant metrics
    for smell, metrics_list in smells_dict.items():
        for metric in metrics_list:
            if metric not in list(df_metrics_new.keys()):
                df_metrics_new[metric] = df_metrics[metric]

    df_metrics_new['IsAbstract'] = df_metrics['IsAbstract']
    df_metrics_new['NumberOfPublicMethods_bool'] = df_metrics['NumberOfPublicMethods_bool']

    # normalize the data after z_score to be between 0 to 1
    df_metrics_new[metrics_to_normalize] = df_metrics_new.groupby(['Project', 'Version'])[
        metrics_to_normalize].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()))

    # add the calculated code smells to the new and final dataframe
    for smells in smells_list:
        df_metrics_new[smells] = df_metrics[smells]

    # create a dictionary of the projects and it's versions
    project_version_dict = df_metrics_new.groupby('Project')['Version'].unique().apply(lambda x: sorted(x)).to_dict()

    # split the data to train and test base on the versions
    df_train, df_test = first_split(df_metrics_new, project_version_dict)

    df_train.fillna(0, inplace=True)
    df_test.fillna(0, inplace=True)

    if to_save:
        df_train.to_csv('df_train.csv', index=False)
        df_test.to_csv('df_test.csv', index=False)

    return df_train, df_test