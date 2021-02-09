import geopandas as gpd
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm, Normalize
from io import StringIO
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz 
from IPython.display import Image, display, Markdown
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.base import clone
from sklearn.inspection import permutation_importance, partial_dependence


# Data structures
################################################################################

# GeoDataFrame containing the map of the world
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


# Functions
################################################################################

def MakeList(x):
    if type(x) == list:
        return x
    return [x]



def ArrangePlots(nplots, nrows, ncols, figsize, xsize=5, ysize=3.5):
    if nrows == None:
        if nplots < ncols:
            ncols = nplots
            nrows = 1
        elif nplots % ncols > 0:
            nrows = nplots//ncols+1
        else:
            nrows = nplots//ncols
    if figsize == None:
        figsize = (ncols*xsize, nrows*ysize)

    return nrows, ncols, figsize



def ImportData(year, resolution, data_dir, land=False, stats=False):
    
    """
    INPUT:
     - year -> Year of data collection. (string)
     - resolution -> Data resolution. (string)
     - data_dir -> Dataset directory. (path string)
     - land -> Choose whether or not to import only data from land hexagons with
             at least one bioclimate variable available. (bool)
     - stats -> Choose whether or not to display some statistics about the data.
             (bool)
    
    OUTPUT:
     - hexagones -> Geodataframe containing all the available selected data. 
    """
    
    # Import hexagon centroids coordinate in a geodataframe
    file_dir = os.path.join(data_dir, 'Spatial',
                   f'Centroids_ISEA3H{resolution}_Geodetic_V_WGS84.shp')
    centroids = gpd.read_file(file_dir)
    
    # Import IGBP land cover class fractions for hexagons in a dataframe
    file_dir = os.path.join(data_dir,
                   f'ISEA3H{resolution}_MCD12Q1_V06_Y{year}_IGBP_Fractions.txt')
    LC = pd.read_csv(file_dir, sep='\t')
    
    # Import bioclimate variables for hexagons in a dataframe
    file_dir = os.path.join(data_dir,
                   f'ISEA3H{resolution}_WorldClim30AS_V02_BIO_Centroid.txt')
    BV = pd.read_csv(file_dir, sep='\t')
    
    # Merge IGBP land cover fractions, bioclimate variables and hexagone coordinates
    hexagons = centroids.merge(LC, on='HID')
    hexagons = hexagons.merge(BV, on='HID')
    n = hexagons.shape[0]
    
    if land:       
        # Identify hexagons composed of water bodies only
        # Water bodies are identified by the fact that they are the ones and only 
        # for which the sum of all land cover class fractions is 0
        LC_rowsums = LC.iloc[:, 1:].sum(axis=1)
        LC_water_idx = LC_rowsums[LC_rowsums == 0].index.values
        nw = len(LC_water_idx)

        # Identify hexagons not only composed of water bodies.
        LC_land_idx = np.array(list(set(range(n))- set(LC_water_idx)))
        nl = len(LC_land_idx)
        
        # Identify hexagons that have no bioclimate variable available
        mask_NA_rows = eval(' & '.join([f'(BV.{col} == -100)'
                                        for col in BV.columns.values[1:]]))    
        BV_NA_idx = BV[mask_NA_rows].index.values
        n_na = len(BV_NA_idx)

        # Identify hexagons that have at least one bioclimate variable available
        BV_A_idx = np.sort(np.array(list(set(BV.index.values)-set(BV_NA_idx))))
        n_a = len(BV_A_idx)

        # Identify hexagons wich satisfy both desired conditions
        land_A_idx = np.intersect1d(BV_A_idx, LC_land_idx)
        nl_a = len(land_A_idx)
        
        # Extract land hexagons with at least one bioclimate variable available.
        hexagons = hexagons.iloc[land_A_idx]
        
        if stats:
            display(Markdown(
                f"The dataset is composed of {n} hexagons, of which:<br>"
                f"- {nl} correspond to lands ({100*nl/n:.4}%);<br>"
                f"- {nw} correspond to water bodies ({100*nw/n:.4}%);<br>"
                f"- {n_a} have at least one bioclimate variable available "
                f"({100*n_a/n:.4}%);<br>"
                f"- {n_na} have no bioclimate variable available "
                f"({100*n_na/n:.4}%);<br>"
                f"- Reducing to the land hexagons:<br>"
                f"-- {nl_a} hexagons have at least one bioclimate variable "
                f"available ({100*nl_a/nl:.4}%)."
            ))
        
        return hexagons
    else:
        if stats:
            display(Markdown(f"The dataset is composed of {n} hexagons."))
                    
        return hexagons
    


def EmpiricalDistribution(data, NA_val=-100, figsize=(15, 8), show=True, save=False,
                          plot_dir='Output/Plots', title='ED', save_params={}):
    """
    INPUT:
     - data -> Data vector (Series or array-like) 
     - NA_val -> Value used for NA.  
     - figsize -> Size of the figure object. (tuple of int: (width, height))
     - show -> Choose whether or not to display the plot. (bool) 
     - save -> Choose whether or not to save the plot. (bool)
     - plot_dir -> Plot saving directory. (path string)
     - title -> Name of the plot file without file extension. (string)
     - save_params -> Parameters for the saving operation. (dict)                
    """

    # Default values for parameter dictionaries:
    # save_params
    SP = {'format': 'jpg'}

    # Update parameter dictionaries with user choices
    SP.update(save_params)

    # Remove NA values if needed
    NA_mask = data != NA_val
    data = data[NA_mask]
    n_NA = (~NA_mask).sum()
    if n_NA > 0:
        display(Markdown(f"There are {n_NA} not avalable data points"))

    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(nrows=2, ncols=2, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])

    # Frequency histogram
    bw = (data.max()-data.min())/100
    sns.histplot(data=data, color='darkblue', binwidth=bw, ax=ax0)
    ax0.set(xlabel='', ylabel='Counts')

    # Density histogram
    sns.kdeplot(x=data, color='darkblue', fill=True, ax=ax1)
    ax1.set(xlabel='', ylabel='Density')

    # Boxplot
    sns.boxplot(x=data, color='skyblue', fliersize=3, ax=ax2)
    ax2.set_xlabel('')

    fig.tight_layout()

    # Save the plot if needed
    if save:
        # Output file directory
        file_dir = os.path.join(plot_dir, f"{title}.{SP['format']}")
        plt.savefig(file_dir, **SP)
    # Prevent display of the plot if needed
    if not show:
        plt.close()
        
        
        
def ConvertScores(GS, cv, convertion):
    
    """
    INPUT:
     - GS -> Dataframe containing sklearn.model_selection.GridSearchCV.cv_results_.
             (dataframe)
     - cv -> Number of folds of k-fold cross-validation performed during the
             grid search. (int)
     - convertion -> Dictionary of score converters. Items 'key: value' are so
             defined:
             - 'key': actual score name (string);
             - 'value': it has the form (new_score, t), where 'new_score' is the
                name of the new score (string) and 't' is the function 
                performing the transformation from the old to the new score 
                (function).         
             
    OUTPUT:
     - df -> Dataframe containing the same information as GS, but with score
             columns converted as in 'convertion'. (dataframe)
    """
    
    df = GS.copy()
    
    for score, (new_score, t) in convertion.items():
        # Rename columns
        df.rename(columns={f'split{f}_test_{score}':
                           f'split{f}_test_{new_score}' for f in range(cv)},
                  inplace=True)
        df.rename(columns={f'mean_test_{score}': f'mean_test_{new_score}',
                           f'std_test_{score}': f'std_test_{new_score}',
                           f'rank_test_{score}': f'rank_test_{new_score}'},
                  inplace=True)
        
        # Apply score convertion
        df[[f'split{f}_test_{new_score}' for f in range(cv)]] = \
        df[[f'split{f}_test_{new_score}' for f in range(cv)]].apply(t)
        df[f'mean_test_{new_score}'] = df[f'mean_test_{new_score}'].apply(t)

    return df



def OneSeRule(GS, parameter, toward_larger_values=True, cv=10, param_grid={},
              mean_error_col='mean_test_score', std_error_col='std_test_score'):
    
    """
    INPUT:
     - GS -> Dataframe containing sklearn.model_selection.GridSearchCV.cv_results_
             or with the same structure and column names. (dataframe)
     - parameter -> Name of the parameter of which the 1-standard-error value 
             must be found. (string)
     - toward_larger_values -> Choose whether the complexity of the resulting  
             model increases as the parameter value grows (True) or as the  
             parameter value decreases (False). (bool)
     - cv -> Number of folds of k-fold cross-validation performed during the
             grid search. (int)
     - param_grid -> Dictionary associating other parameters names (different 
             from the parameter object of the analysis) with single specific 
             values assumed during the grid search in order to restric the 
             research to a subset of the cosidered models. (dict)
     - mean_error_col -> Name of the dataframe column containing the average 
             model error. (string)
     - std_error_col -> Name of the dataframe column containing the standard 
             deviation of the mean model error. (string)          
             
    OUTPUT:
     - dictionary containing the 1-standard-error parameter and the mean error 
             of the corresponding model. (dict) None in case no 1-standard-error
             parameter can be found. 
    """
    
    # Name of the parameter column in the dataframe as in the column structure 
    # of sklearn.model_selection.GridSearchCV.cv_results_ 
    parameter_col = f'param_{parameter}'
    # Find the sub-dataframe identified by param_grid
    if param_grid:
        mask = eval(" & ".join([f"(GS['param_{p}'] == {param_grid[p]})" 
                                for p in param_grid.keys()]))
        R = GS[mask].copy()
    else:
        R = GS.copy()
    
    # Find the row for which the mean model error is minimised
    best_id = R[mean_error_col].idxmin()
    # Extract the parameter value, ...
    best_param = R.loc[[best_id], parameter_col].item()
    # ... the mean model error, ...
    best_error = R.loc[[best_id], mean_error_col].item()
    # ... the standard error of the mean model error
    best_se = R.loc[[best_id], std_error_col].item()/np.sqrt(cv)
    
    # Identify parameter values for which the mean model error is no more than 
    # 1 standard error away from the achieved minimim mean model error 
    if toward_larger_values:
        mask = eval("&".join([f"(R['{parameter_col}'] > best_param)",
                              f"(R['{mean_error_col}'] <= best_error+best_se)"]))
    else:
        mask = eval("&".join([f"(R['{parameter_col}'] < best_param)",
                              f"(R['{mean_error_col}'] <= best_error+best_se)"]))    
    R_SE = R[mask]
    
    # If no parameter value satisfy the condition
    if R_SE.empty:
        print(f"No 1-standard-error {parameter} can be obtained. "
              "None value has been returned instead.")
        return None
    # Otherwise, identify the 1-standard-error parameter value
    else:
        if toward_larger_values:
            f = 'max'
        else:
            f = 'min'
        mask = eval(f"(R_SE['{parameter_col}'] == R_SE['{parameter_col}'].{f}())")
        best_1se_param = R_SE[mask][parameter_col].item()
        best_1se_error = R_SE[mask][mean_error_col].item()  

        return {parameter: best_1se_param, 'error': best_1se_error}

    
    
def GridSearchWarmStart(X, y, estimator, param_grid,
                        scoring={'R2E': lambda y_true, y_pred: 1-r2(y_true=y_true, y_pred=y_pred)},
                        cv=10, oob_score=True):
    
    """
    INPUT:
     - X -> Feature matrix. (ndarray or DataFrame)
     - y -> Target array. (array-like)
     - estimator -> A sklearn random forest estimator object. (object) 
     -  param_grid -> Dictionary with parameters names (string) as keys and 
             lists of parameter settings to try as values. (dict)
     - scoring -> Dictionary with score names (string) as keys and functions to 
             compute them as values. (dict) 
     - cv -> Number of folds of k-fold cross-validation to perform during the
             grid search. (int) If None, no k-fold cross-validation is performed.
     - oob_score -> Choose whether of not the oob score (R2E) should be 
             computed. (bool)
    
    OUTPUT:
     - GS -> Results of the grid search. (DataFrame)
    """
    
    
    
    # Names of the explored hyperparameters. 
    param_cols = list(param_grid.keys())

    # Number of n_estimators values explored
    if 'n_estimators' in param_grid:
        N = len(param_grid['n_estimators'])
        # Move n_estimators at the end of the list 
        # (it is needed for sorting dataframe values in the next steps) 
        param_cols.append(param_cols.pop(param_cols.index('n_estimators')))
        # Count all combinations of hyperparameters values excluding n_estimators
        # If only n_estimators are explored 
        if len(param_cols[:-1]) == 0: 
            count = 1   
        # If additional hyperparameters are explored
        else: 
            count = np.prod([len(param_grid[c]) for c in param_cols[:-1]]) 
    else: 
        N = 1
        count = np.prod([len(param_grid[c]) for c in param_cols])
        
    # Dataframe that will contain hyperparameter combinations and 
    # model performances in the grid search
    GS = pd.DataFrame(list(ParameterGrid(param_grid)))
    GS.sort_values(by=param_cols, inplace=True, ignore_index=True)
    
    # Use Out-Of-Bag samples to estimate model performance
    if oob_score:
        print(f"Grid Search with Out-Of-Bag samples.\n"
              f"Fitting {GS.shape[0]} candidates.\n")
        # Column that will contain the OOB score
        GS['OOB_R2E'] = np.nan 
        
        # Grid-Search
        # Cycle through every combination of hyperparameter values,
        # except n_estimators
        for i in range(count):
            # Base estimator
            rgr = clone(estimator)
            rgr.set_params(oob_score=oob_score)
            # Cycle through every n_estimators value
            for i_n in range(N):
                params = {p: GS.loc[i*N+i_n, p] for p in param_cols}
                # Update n_estimators and set the other parameters.
                # For every i, all parameters but n_estimators are fixed,
                # so here we set the parameters when i_n=0, but for i_n>0
                # we update n_estimators only
                rgr.set_params(**params)
                # Fitting the Random Forest
                rgr.fit(X, y)
                # OOB error
                GS.at[i*N+i_n,'OOB_R2E'] = 1-rgr.oob_score_
            print(f"   {(i+1)*N} candidates explored.")
        print()
    
    # Use k-fold cross-validation to estimate model performance
    if pd.notna(cv):
        print(f"Grid Search with k-fold Cross-Validation.\n"
              f"Fitting {cv} folds for each one of the {GS.shape[0]} candidates,"
              f" totalling {cv*GS.shape[0]} fits.\n")
    
        # Dataframe columns that will be filled during the grid search
        cols = []
        for m in scoring.keys():
            cols.extend([f'split{i}_test_{m}' for i in range(cv)])
            cols.extend([f'{l}_test_{m}' for l in ['mean','std']])
        GS[cols] = np.nan 
        
        # Splitting the training set into folds 
        folds = [f.sample(frac=1) for f in np.array_split(X, cv)]
        
        # Grid-Search
        # Cycle through each fold
        for f in range(cv):
            # Train set: Everything but the selected fold
            train = folds.copy()
            del train[f]
            X_train = pd.concat(train, sort=False)
            y_train = y.loc[X_train.index]
            # Test set: Selected fold
            Xtest = folds[f]
            ytest = y.loc[Xtest.index]
            
            # Cycle through every combination of hyperparameter values, 
            # except n_estimators
            for i in range(count):
                # Base Random Forest
                rgr = clone(estimator)
                # Cycle through every n_estimators value
                for i_n in range(N):
                    params = {p: GS.loc[i*N+i_n, p] for p in param_cols}
                    # Update n_estimators and set the other parameters.
                    # For every i, all parameter but n_estimators are fixed,
                    # so here we set the parameters when i_n=0, but for i_n>0
                    # we update n_estimators only
                    rgr.set_params(**params)
                    # Fitting the Random Forest
                    rgr.fit(X_train, y_train)
                    # Model score performances
                    ypred = rgr.predict(Xtest)
                    for m in scoring.keys():
                        GS.at[i*N+i_n, f'split{f}_test_{m}'] =\
                        scoring[m](y_true=ytest, y_pred=ypred)
            print(f"Fold {f+1}: {GS.shape[0]} fits done.")
        
        for m in scoring.keys():
            cols = [f'split{i}_test_{m}' for i in range(cv)]
            GS[f'mean_test_{m}'] = GS[cols].mean(axis=1)
            GS[f'std_test_{m}'] = GS[cols].std(axis=1)
        
    # Rename parameters columns like in sklearn.model_selection.GridSearchCV
    GS.rename(columns={key: f'param_{key}' for key in param_grid.keys()},
              inplace=True) 
    
    print("Grid Search performed. \n")
    
    return GS



def RemoveAntarctica(GDF):
    """
    INPUT:
     - GDF -> GeoDataFrame with geometry column and other data columns. 
             (GeoDataFrame)
     
    OUTPUT:
     - The same GeoDataFrame with no data point with y geometry coordinate below
             -60 latitude degrees, hence in the Antartic continent. (GeoDataFrame)
    """
    
    return GDF[GDF.geometry.y > -60].copy()



def Whiskers(data):
    """
    INPUT:
     - data -> Data values. (Series)

     OUTPUT:
      - min_w, max_w -> minimum and maximum whisker of the input data. (floats)
    """

    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    IQR = q3-q1

    max_w = q3 + 1.5*IQR
    min_w = q1 - 1.5*IQR

    if min_w < data.min():
        min_w = data.min()
    if max_w > data.max():
        max_w = data.max()

    return min_w, max_w



def GeoPlot(GDF, col, ax=None, figsize=(50, 27), Antarctica=True,
            show=True, save=False, plot_dir='Output/Plots', title='GeoPlot',
            save_params={}, plot_params={}, adjust_colorbar_params={}):
    
    """
    INPUT:
     - GDF -> GeoDataFrame with geometry column and other data columns. 
             (GeoDataFrame)
     - col -> Name of the column containing data to plot. (string)
     - ax -> Axes of the current figure. (matplotlib axes object)
     - figsize -> Size of the figure object. (tuple of int: (width, height))
     - Antarctica -> Choose whether or not to plot the Antartic continent. (bool)
     - show -> Choose whether or not to display the plot. (bool) 
     - save -> Choose whether or not to save the plot. (bool)
     - plot_dir -> Plot saving directory. (path string)
     - title -> Name of the plot file without file extension. (string)
     - save_params -> Parameters for the saving operation. (dict)
     - plot_params -> Parameters for geopandas.GeoDataFrame.plot. (dict)
             Look at the doc. for more info. 
             Three additional items are included:
             - boundary_color -> Countries' borders color. (string)
             - land_color -> Lands color. (string)
             - sea_color -> Sea color. (string)      
      - adjust_colorbar_params -> Parameters used to adjust the colorbar 
              settings. Items in the dictionary are:
              - center_on_zero -> Choose whether or not to center the colorbar 
                  on the value 0. (bool) Make sure to choose in plot_params a 
                  diverging colormap to obtain a better outcome.
              - remove_outliers -> Choose whether or not to remove outliers from
                  the colorbar range of values. Outliers will be colored with 
                  the same color of the closest value on the colorbar. 
              - vmin, vmax -> Minimum and maximum values considered in the 
                  colorbar's range of values. (int) Smaller or larger values are
                  represented with same color of the closest value on the 
                  colorbar.
              - labelsize -> Size of the colorbar labels. (int)
              - valueson -> Specific values that will be represented on the 
                  colorbar as straigth vertical lines. (float or list of floats)
              - valueson_color -> Color of lines on the colorbar. (string)       
              - valueson_linewidth -> Width of lines on the colorbar. (int)                      
    """
    
    # Default values for parameter dictionaries:
    # plot_params
    PP = {'cmap': 'viridis', 'marker': 'h', 'markersize': 9, 
          'boundary_color': 'white', 'land_color': 'whitesmoke', 
          'sea_color': '#adcfeb', 'legend': True,
          'legend_kwds': {'orientation': 'horizontal'}}
    # adjust_colorbar_params
    ACP = {'center_on_zero': False, 'remove_outliers': False, 'vmin': None,
           'vmax': None, 'labelsize': 40, 'valueson': [],
           'valueson_color': 'yellow', 'valueson_linewidth': 5}
    # save_params
    SP = {'format': 'jpg'}

    # Update parameter dictionaries with user choices
    PP.update(plot_params)
    ACP.update(adjust_colorbar_params)
    SP.update(save_params)

    # Output file directory
    file_dir = os.path.join(plot_dir, f"{title}.{SP['format']}")

    # If the plot file already exists and must not be overwritten, then display it.
    if show and os.path.exists(file_dir) and not save:
        if ax is None:
            display(Image(filename=file_dir, retina=True))
        else:
            ax.imshow(plt.imread(file_dir), aspect='equal')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    # If the file must be created or overwritten ...
    else:
        # Import the geodataframe containing the map of the world
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        # Remove Antarctica if needed
        if not Antarctica:
            world = world[(world.name != "Antarctica")]
            GDF = RemoveAntarctica(GDF)

        # Remove outliers from the colorbarif needed
        if ACP['remove_outliers']:
            min_w, max_w = Whiskers(GDF[col])
            if ACP['vmin'] is None:
                ACP['vmin'] = min_w
            if ACP['vmax'] is None:
                ACP['vmax'] = max_w
        else:
            if ACP['vmin'] is None:
                ACP['vmin'] = GDF[col].min()
            if ACP['vmax'] is None:
                ACP['vmax'] = GDF[col].max()
        # Center the colorbar on 0 if needed
        if ACP['center_on_zero']:
            PP['norm'] = TwoSlopeNorm(vmin=ACP['vmin'], vcenter=0,
                                      vmax=ACP['vmax'])
        else:
            PP['norm'] = Normalize(vmin=ACP['vmin'], vmax=ACP['vmax'])

        with sns.axes_style("white"):
            # Create the plot in the given axes or in a new axes
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)

            # Legend adjustments if colorbar must be displayed
            if PP['legend']:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("bottom", size="5%", pad=0.1)
                cax.tick_params(labelsize=ACP['labelsize'])
                PP['cax'] = cax

            # Plot of the world on the backgroud
            world.plot(ax=ax, facecolor=PP.pop('land_color'), edgecolor="none",
                       zorder=1).set_facecolor(PP.pop('sea_color'))
            # Plot of the countries' borders on the foreground
            world.boundary.plot(ax=ax, color=PP.pop('boundary_color'), zorder=3)

            # Plot of the data points
            GDF.plot(column=col, ax=ax, **PP, zorder=2)

            # Hide axis
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Draw lines on the colorbar on the specified values
            if PP['legend']:
                ACP['valueson'] = MakeList(ACP['valueson'])
                for v in ACP['valueson']:
                    cax.vlines(v, -1, 1, colors=ACP['valueson_color'],
                               linewidth=ACP['valueson_linewidth'])

            # fig.tight_layout()

            # Save the plot if needed
            if save:
                plt.savefig(file_dir, **SP)
            # Prevent display of the plot if needed
            if not show:
                plt.close()

                
                
def VisualiseTree(tree, max_depth, feature_labels=None,
                  show=True, save=False, plot_dir='Output/Plots',
                  title='DTR', plot_format='png'):
    """
    INPUT:
     - tree -> A sklearn fitted tree object. (object)
     - max_depth -> Maximum depth of the tree that will be visualised. (int)
     - feature_labels -> Labels of the model features that will be used in the 
             plot. (list of strings).
     - show -> Choose whether or not to display the plot. (bool) 
     - save -> Choose whether or not to save the plot. (bool)
     - plot_dir -> Plot saving directory. (path string)
     - title -> Name of the plot file without file extension. (string)
     - plot_format -> Format of the plot file. (string)
    """

    # Output file directory
    file_dir = os.path.join(plot_dir, f"{title}.{plot_format}")

    # If the plot file already exists and must not be overwritten, then display it.
    if show and os.path.exists(file_dir) and not save:
        display(Image(filename=file_dir, retina=True))

    # If the file must be created or overwritten ...
    else:
        # Create an empty object to receive a string buffer in DOT format.
        dot_data = StringIO()
        # Exports the tree in DOT format into the out_file
        export_graphviz(tree, out_file=dot_data, max_depth=max_depth,
                        feature_names=feature_labels, proportion=True,
                        filled=True, rounded=True)
        # Use the DOT object to create the graph.
        tree = graph_from_dot_data(dot_data.getvalue())

        # Save the plot if needed
        if save:
            tree.write(file_dir, format=plot_format)
        # Display the plot if needed
        if show:
            display(Image(tree.create(format=plot_format), retina=True))
            
            
            
def FeatureImportanceRanking(estimator, feature_labels, 
                             ax=None, figsize=(10, 10), rounding=3,
                             show=True, save=False, plot_dir='Output/Plots',
                             title='FIR', save_params={}, plot_params={}):
    """
    INPUT:
     - estimator -> A sklearn tree-based fitted estimator object that ows the 
             attribute feature_importances. (object)
     - feature_labels -> Labels of t he model features that will be used in the 
             plot. (list of strings).
     - ax -> Axes of the current figure. (matplotlib axes object)
     - figsize -> Size of the figure object. (tuple of int: (width, height))
     - rounding -> Number of decimals for labels. (int)
     - show -> Choose whether or not to display the plot. (bool) 
     - save -> Choose whether or not to save the plot. (bool)
     - plot_dir -> Plot saving directory. (path string)
     - title -> Name of the plot file without file extension. (string)
     - save_params -> Parameters for the saving operation. (dict)
     - plot_params -> Parameters for pandas.DataFrame.plot. (dict)
             Look at the doc. for more info. 
    """
    
    # Default values for parameter dictionaries:
    # save_params
    SP = {'format': 'jpg'}
    # plot_params
    PP = {'color': 'darkcyan', 'legend': False, 'fontsize': 15, 'width': 0.9}

    # Update parameter dictionaries with user choices
    SP.update(save_params)
    PP.update(plot_params)

    # Output file directory
    file_dir = os.path.join(plot_dir, f"{title}.{SP['format']}")

    # If the plot file already exists and must not be overwritten, then display it.
    if show and os.path.exists(file_dir) and not save:
        if ax == None:
            display(Image(filename=file_dir, retina=True))
        else:
            ax.imshow(plt.imread(file_dir), aspect='equal')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    
    # If the file must be created or overwritten ...
    else:
        # Create a Series containing the feature importance values
        FI = pd.Series(estimator.feature_importances_, index=feature_labels)\
            .sort_values()

        # Create the plot in the given axes or in a new axes
        if ax == None:
            fig, ax = plt.subplots(figsize=figsize)

        FI.plot(kind='barh', ax=ax, **PP)
        ax.get_xaxis().set_visible(False)
        ax.set(facecolor='white')
        for i, imp in enumerate(FI):
            ax.text(imp+0.001, i-0.13, str(round(imp, rounding)))
        
        # Save the plot if needed
        if save:
            plt.savefig(file_dir, **SP)
        # Prevent display of the plot if needed
        if not show:
            plt.close()
            
            
            
def PermutationImportanceRanking(estimator, X, y, feature_labels,
                                 ax=None, figsize=(10, 10), rounding=3,
                                 show=True, save=False, plot_dir='Output/Plots',
                                 title='PIR', save_params={}, pi_params={},
                                 plot_params={}):
    """
    INPUT:
     - estimator -> A sklearn tree-based fitted estimator object. (object) 
             Look at sklearn.inspection.permutation_importance doc. for more info. 
     - X -> Feature matrix. (ndarray or DataFrame)
             Look at sklearn.inspection.permutation_importance doc. for more info.
     - y -> Target array. (array-like or None)
             Look at sklearn.inspection.permutation_importance doc. for more info.
     - feature_labels -> Labels of the model features that will be used in the 
             plot. (list of strings).
     - ax -> Axes of the current figure. (matplotlib axes object)
     - figsize -> Size of the figure object. (tuple of int: (width, height))
     - rounding -> Number of decimals for labels. (int)
     - show -> Choose whether or not to display the plot. (bool) 
     - save -> Choose whether or not to save the plot. (bool)
     - plot_dir -> Plot saving directory. (path string)
     - title -> Name of the plot file without file extension. (string)
     - save_params -> Parameters for the saving operation. (dict)
     - pi_params -> Parameters for sklearn.inspection.permutation_importance. 
             (dict) Look at the doc. for more info.
     - plot_params -> Parameters for pandas.DataFrame.plot. (dict)
             Look at the doc. for more info. 
    """

    # Default values for parameter dictionaries:
    # save_params
    SP = {'format': 'jpg'}
    # pi_params
    PIP = {'scoring': 'neg_mean_squared_error', 'n_repeats': 10, 'n_jobs': 1,
           'random_state': 1}
    # plot_params
    PP = {'color': 'goldenrod', 'legend': False, 'capsize': 3, 'fontsize': 15,
          'width': 0.9}

    # Update parameter dictionaries with user choices
    SP.update(save_params)
    PIP.update(pi_params)
    PP.update(plot_params)

    # Output file directory
    file_dir = os.path.join(plot_dir, f"{title}.{SP['format']}")

    # If the plot file already exists and must not be overwritten, then display it.
    if show and os.path.exists(file_dir) and not save:
        if ax == None:
            display(Image(filename=file_dir, retina=True))
        else:
            ax.imshow(plt.imread(file_dir), aspect='equal')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    # If the file must be created or overwritten ...
    else:
        # Compute permutation importance values
        PI = permutation_importance(estimator=estimator, X=X, y=y, **PIP)
        # Create a dataframe with means and standard deviations only.
        del PI['importances']
        PI = pd.DataFrame(PI, index=feature_labels)\
            .sort_values(by='importances_mean')

        # Create the plot in the given axes or in a new axes
        if ax == None:
            fig, ax = plt.subplots(figsize=figsize)

        PI.plot(y='importances_mean', kind='barh', xerr=PI['importances_std'],
                ax=ax, **PP)
        ax.get_xaxis().set_visible(False)
        ax.set(facecolor='white')
        for i, (imp, se) in enumerate(zip(PI['importances_mean'],
                                          PI['importances_std'])):
            ax.text(imp+se+0.005*PI['importances_mean'][-1], i-0.13,
                    str(round(imp, rounding)))
        # Save the plot if needed
        if save:
            plt.savefig(file_dir, **SP)
        # Prevent display of the plot if needed
        if not show:
            plt.close()            
            
            
def PartialDependencePlots(estimator, X, features, feature_labels,
                           nrows=None, ncols=4, figsize=None, sharey=True,
                           conf_int=True,
                           show=True, save=False, plot_dir='Output/Plots',
                           title='PDP', save_params={}, pdp_params={},
                           plot_params={}, plot_ci_params={}):
    """
    INPUT:
     - estimator -> A sklearn tree-based fitted estimator object. (object) 
             Look at sklearn.inspection.partial_dependence doc. for more info.
     - X -> Feature matrix. (array-like or dataframe)
             Look at sklearn.inspection.partial_dependence doc. for more info.
     - features -> Features for which the partial dependency should be computed.
             (int, string or list of ints, strings)
             Look at sklearn.inspection.partial_dependence doc. for more info.
     - feature_labels -> Labels of the model features that will be used in the 
             plot. (string or list of strings).            
     - nrows -> Number of rows of the figure object. (int)
     - ncols -> Number of columns of the figure object. (int)
     - figsize -> Size of the figure object. (tuple of int: (width, height))
     - sharey -> Choose whether or not axes in the figure should share the y 
             axis values. (bool) 
     - conf_int -> Choose whether or not to plot the confidence interval. (bool) 
     - show -> Choose whether or not to display the plot. (bool) 
     - save -> Choose whether or not to save the plot. (bool)
     - plot_dir -> Plot saving directory. (path string)
     - title -> Name of the plot file without file extension. (string)
     - save_params -> Parameters for the saving operation. (dict)
     - pdp_params -> Parameters for sklearn.inspection.partial_dependence. (dict)
             Look at the doc. for more info.
     - plot_params -> Parameters for matplotlib.pylot.plot. (dict)
             Look at the doc. for more info.
     - plot_ci_params -> Parameters for matplotlib.axes.Axes.fill_between. (dict)
             Look at the doc. for more info.   
             
    OUTPUT:
     - fig -> Figure object. (matplotlib figure object)
     - ax, axs -> Axes of the current figure. (matplotlib axes object)
    """

    # Default values for parameter dictionaries:
    # save_params
    SP = {'format': 'jpg'}
    # pdp_params
    PDPP = {'kind': 'both', 'grid_resolution': 100}
    # plot_params
    PP = {}
    # plot_ci_params
    PCIP = {'alpha': 0.2, 'color': '#66C2D7'}

    # Update parameter dictionaries with user choices
    SP.update(save_params)
    PDPP.update(pdp_params)
    PP.update(plot_params)
    PCIP.update(plot_ci_params)

    # If features and feature_labels contains only a string, make them lists.
    features = MakeList(features)
    feature_labels = MakeList(feature_labels)
    n = len(features)
    
    # Define number of rows and columns and figsize of the figure object
    # containing the plot(s)
    nrows, ncols, figsize = ArrangePlots(n, nrows, ncols, figsize)

    # Output file directory
    file_dir = os.path.join(plot_dir, f"{title}.{SP['format']}")

    # If the plot file already exists and must not be overwritten, then display it.
    if show and os.path.exists(file_dir) and not save:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(plt.imread(file_dir), aspect='equal')
        plt.axis('off')

        return fig, ax
    
    # If the file must be created or overwritten ...
    else:
        fig, axs = plt.subplots(nrows, ncols, sharey=sharey, figsize=figsize)

        # Go through each feature
        for f, l, ax in zip(features, feature_labels, np.array(axs).flat):
            # Compute partial dependence values
            PDP = partial_dependence(estimator=estimator, X=X, features=f, 
                                     **PDPP)

            ax.plot(PDP['values'][0], PDP['average'][0], **PP)

            if 'individual' in PDP and conf_int:
                # Compute the standard error on each mean partial dependence
                PDP['sd'] = PDP['individual'][0].std(axis=0).reshape(1, -1)
                # Define upper and lower bounds for the confidence interval
                upper = PDP['average'][0]+PDP['sd'][0]
                lower = PDP['average'][0]-PDP['sd'][0]

                ax.fill_between(PDP['values'][0], upper, lower, **PCIP)

            ax.set_xlabel(l)
            if ax.is_first_col():
                ax.set_ylabel('Target')

        # Remove eventual excessive axes
        if n < nrows*ncols:
            for i in range(1, nrows*ncols-n+1):
                fig.delaxes(axs.flat[-i])

        fig.tight_layout()

        # Save the plot if needed
        if save:
            plt.savefig(file_dir, **SP)
        # Prevent display of the plot if needed
        if not show:
            plt.close()

        return fig, axs
    
    
    
def ShapSummary(estimator, X, feature_labels=None,
                show=True, save=False, plot_dir='Output/Plots',
                title='ShapSummary', save_params={}, plot_params={}):
    """
    INPUT:
     - estimator -> A tree-based fitted estimator object. (object)
             Look at the item 'model' in shap.TreeExplainer doc. for more info.  
     - X -> Feature matrix. (numpy.array or pandas.DataFrame of list)
             Look at the shap.TreeExplainer.shap_values doc and at the item 
             'features' in the shap.summary_plot doc. for more info. 
     - feature_labels -> Labels of the model features that will be used in the 
             plot. (string or list of strings). Look at the item 'feature_names'
             in the shap.summary_plot doc. for more info.        
     - show -> Choose whether or not to display the plot. (bool) 
     - save -> Choose whether or not to save the plot. (bool)
     - plot_dir -> Plot saving directory. (path string)
     - title -> Name of the plot file without file extension. (string)
     - save_params -> Parameters for the saving operation. (dict)
     - plot_params -> Parameters for shap.summary_plot. (dict)
             Look at the doc. for more info.
    """    

    # Default values for parameter dictionaries:
    # save_params
    SP = {'format': 'jpg'}
    # plot_params
    PP = {'plot_type': 'dot', 'cmap': 'magma', 'alpha': 0.5, 
          'plot_size': (15, 12)}

    # Update parameter dictionaries with user choices
    SP.update(save_params)
    PP.update(plot_params)

    # Output file directory
    file_dir = os.path.join(plot_dir, f"{title}.{SP['format']}")

    # If the plot file already exists and must not be overwritten, then display it.
    if show and os.path.exists(file_dir) and not save:
        fig, ax = plt.subplots(figsize=PP['plot_size'])
        ax.imshow(plt.imread(file_dir), aspect='equal')
        plt.axis('off')

    # If the file must be created or overwritten ...
    else:
        # Create the object that can calculate shap values
        explainer = shap.TreeExplainer(model=estimator)
        # Calculate shap values
        shap_values = explainer.shap_values(X)

        shap.summary_plot(shap_values, features=X, feature_names=feature_labels,
                          show=False, **PP)
        plt.tight_layout()

        # Save the plot if needed
        if save:
            plt.savefig(file_dir, **SP)
        # Prevent display of the plot if needed
        if not show:
            plt.close()
            
            
            
def ShapDependencePlots(estimator, X, features, feature_labels,
                        interaction_index=None, colorbar_labels=None,
                        nrows=None, ncols=4, figsize=None, sharey=True,
                        show=True, save=False, plot_dir='Output/Plots',
                        title='SDP', save_params={}, plot_params={}):
    
    """
    INPUT:
     - estimator -> A tree-based fitted estimator object. (object)
             Look at the item 'model' in shap.TreeExplainer doc. for more info.  
     - X -> Feature matrix. (numpy.array or pandas.DataFrame of list)
             Look at the shap.TreeExplainer.shap_values doc and at the item 
             'features' in the shap.dependence_plot doc. for more info. 
     - features -> Features for which the shap dependency should be computed.
             (int, string or list of ints, strings)
             Look at shap.dependence_plot doc. for more info.    
     - feature_labels -> Labels of the model features that will be used in the 
             plot. (string or list of strings). Look at the item 'feature_names'
             in the shap.dependence_plot doc. for more info.  
     - interaction_index -> The index of the feature used to color the 
             dependence plot. It can takes values:
             - a string: it must be the name of a feature;
             - a int: it must be the numeric index of a feature;
             - a dictionary: keys and values must be feature names and in each 
               pair (key, value), 'value' represents the feature used to color
               the dependence plot of feature 'key';
             - 'auto': shap.common.approximate_interactions is used to pick what
               seems to be the strongest interaction;
             - None: no feature is used to color the dependence plot.
     - colorbar_labels -> List of feature labels to use next to colorbar if
             interaction_index is not None. (string or list of strings)  
     - nrows -> Number of rows of the figure object. (int)
     - ncols -> Number of columns of the figure object. (int)
     - figsize -> Size of the figure object. (tuple of int: (width, height))
     - sharey -> Choose whether or not axes in the figure should share the y 
             axis values. (bool) 
     - show -> Choose whether or not to display the plot. (bool) 
     - save -> Choose whether or not to save the plot. (bool)
     - plot_dir -> Plot saving directory. (path string)
     - title -> Name of the plot file without file extension. (string)
     - save_params -> Parameters for the saving operation. (dict)
     - plot_params -> Parameters for shap.dependence_plot. (dict)
             Look at the doc. for more info.
    """  

    # Default values for parameter dictionaries:
    # save_params
    SP = {'format': 'jpg'}
    # plot_params
    PP = {'color': '#006080', 'cmap': 'magma', 'dot_size': 16, 'alpha': 0.5}

    # Update parameter dictionaries with user choices
    SP.update(save_params)
    PP.update(plot_params)

    # If features contains only a string, make it a list.
    features = MakeList(features)
    n = len(features)

    # Define number of rows and columns and figsize of the figure object
    # containing the plot(s)
    nrows, ncols, figsize = ArrangePlots(n, nrows, ncols, figsize)

    # Output file directory
    file_dir = os.path.join(plot_dir, f"{title}.{SP['format']}")

    # If the plot file already exists and must not be overwritten, then display it.
    if show and os.path.exists(file_dir) and not save:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(plt.imread(file_dir), aspect='equal')
        plt.axis('off')

        return fig, ax

    else:
        # Dictionary associating feature label to colorbar label.
        # If the colorbar labels are not given, it uses the variable index
        if colorbar_labels == None:
            colorbar_labels = dict(zip(feature_labels,
                              [f'v{i}' for i in range(len(feature_labels))]))
        else:
            colorbar_labels = dict(zip(feature_labels, colorbar_labels))
        
        # Dictionary associating each feature to its interaction index 
        if type(interaction_index) == dict:
            II = interaction_index
        else:
            II = dict(zip(features, [interaction_index]*n))

        fig, axs = plt.subplots(nrows, ncols, sharey=sharey, figsize=figsize)

        # Go through each feature
        for f, (i, ax) in zip(features, enumerate(np.array(axs).flat)):
            # Create the object that can calculate shap values
            explainer = shap.TreeExplainer(model=estimator)
            # Calculate shap values
            shap_values = explainer.shap_values(X)

            shap.dependence_plot(f, shap_values, features=X, ax=ax, show=False,
                                 feature_names=feature_labels, **PP,
                                 interaction_index=II[f])

            ax.tick_params(axis='both', left=False, bottom=False)
            ax.spines['left'].set_linewidth(0)
            ax.spines['bottom'].set_linewidth(0)
            if i % ncols == 0:
                ax.set_ylabel('SHAP value')
            else:
                ax.set_ylabel('')
                ax.tick_params(axis='y', labelleft=False)

        # Remove eventual excessive axes
        if n < nrows*ncols:
            for i in range(1, nrows*ncols-n+1):
                fig.delaxes(axs.flat[-i])

        # Update colorbar labels if needed
        if interaction_index != None:
            for i in range(1, n+1):
                lab = fig.axes[-i].get_ylabel()
                fig.axes[-i].set(ylabel='', xlabel=colorbar_labels[lab])
                fig.axes[-i].xaxis.set_label_position('top')

        fig.tight_layout()

        # Save the plot if needed
        if save:
            plt.savefig(file_dir, **SP)
        # Prevent display of the plot if needed
        if not show:
            plt.close()

        return fig, axs
    
