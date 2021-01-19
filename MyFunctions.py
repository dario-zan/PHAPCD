import geopandas as gpd
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm, Normalize
from io import StringIO
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz 
from IPython.display import Image, display
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.base import clone

# Data Structures
#################

from MyConstants import bv_cols




# GeoDataFrame containing the map of the world
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


# Functions
###########

def ImportLandData(year, res, data_dir, centroids_dir, stats = True):
    # Import hexagon centroids coordinate in a geodataframe
    centroids = gpd.read_file(os.path.join(data_dir, centroids_dir, f'Centroids_ISEA3H{res}_Geodetic_V_WGS84.shp'))

    # Import IGBP land cover class fractions for hexagons in a dataframe
    IGBP_lc = pd.read_csv(os.path.join(data_dir, f'ISEA3H{res}_MCD12Q1_V06_Y{year}_IGBP_Fractions.txt'), sep = '\t')
    # Let identify hexagons only composed of water bodies.
    ## They are the ones for which the sum of all land cover class fractions is 0
    IGBP_rowsums = IGBP_lc.iloc[:,1:].sum(axis=1)
    IGBP_water_idx = IGBP_rowsums[IGBP_rowsums == 0].index.values
    # Let identify hexagons not only composed of water bodies.
    IGBP_land_idx = np.array(list(set(range(centroids.shape[0]))-set(IGBP_water_idx)))
    if stats:
        print(f"There are {centroids.shape[0]} hexagons:\n"
              f" - {len(IGBP_land_idx)} hexagons correspond to lands ({100*len(IGBP_land_idx)/centroids.shape[0]:.4}%);\n"
              f" - {len(IGBP_water_idx)} hexagons correspond to water bodies ({100*len(IGBP_water_idx)/centroids.shape[0]:.4}%);")

    # Import bioclimate variables for hexagons in a dataframe
    BIO_vars = pd.read_csv(os.path.join(data_dir, f'ISEA3H{res}_WorldClim30AS_V02_BIO_Centroid.txt'), sep = '\t')
    # Let identify hexagons that have none bioclimate variable available
    mask_NA_rows = eval(' & '.join([f'(BIO_vars.{col} == -100)' for col in BIO_vars.columns.values[1:]]))
    BIO_NA_idx = BIO_vars[mask_NA_rows].index.values
    # Let identify hexagons that have at least one bioclimate variable available
    BIO_A_idx = np.sort(np.array(list(set(BIO_vars.index.values)-set(BIO_NA_idx))))
    # Let identify hexagons wich satisfy both desired conditions
    land_A_idx = np.intersect1d(BIO_A_idx, IGBP_land_idx)
    if stats:
        print(f" - {len(BIO_A_idx)} hexagons have at least one bioclimate variable available ({100*len(BIO_A_idx)/centroids.shape[0]:.4}%);\n"
              f" - {len(BIO_NA_idx)} hexagons have none bioclimate variable available ({100*len(BIO_NA_idx)/centroids.shape[0]:.4}%);\n"
              f" - Reducing to the land hexagons:\n"
              f"   -- {len(land_A_idx)} hexagons have at least one bioclimate variable available ({100*len(land_A_idx)/len(IGBP_land_idx):.4}%).")

    # Let add IGBP land cover fractions and bioclimate variables as attributes to the hexagone coordinates
    hexagons = centroids.merge(IGBP_lc, on='HID')
    hexagons = hexagons.merge(BIO_vars, on='HID')
    # Let restrict to the land hexagons which has at least one bioclimate variable available.
    hexagons = hexagons.iloc[land_A_idx]

    return hexagons

def ImportData(year, res, data_dir, centroids_dir):
    # Import hexagon centroids coordinate in a geodataframe
    centroids = gpd.read_file(os.path.join(data_dir, centroids_dir, f'Centroids_ISEA3H{res}_Geodetic_V_WGS84.shp'))

    # Import IGBP land cover class fractions for hexagons in a dataframe
    IGBP_lc = pd.read_csv(os.path.join(data_dir, f'ISEA3H{res}_MCD12Q1_V06_Y{year}_IGBP_Fractions.txt'), sep = '\t')

    # Import bioclimate variables for hexagons in a dataframe
    BIO_vars = pd.read_csv(os.path.join(data_dir, f'ISEA3H{res}_WorldClim30AS_V02_BIO_Centroid.txt'), sep = '\t')

    # Let add IGBP land cover fractions and bioclimate variables as attributes to the hexagone coordinates
    hexagons = centroids.merge(IGBP_lc, on='HID')
    hexagons = hexagons.merge(BIO_vars, on='HID')

    return hexagons

def EnlargeOuterWindow(data, default_window_size, cleft, cright,
                       coord, eps, tol):
    """
    INPUT:
     - data -> GeoDataFrame with geometry column.
     - default_window_size -> desired size of the window in terms of points contained. (int)
     - cleft, cright -> bounds of the considered coordinate. (float)
     - coord -> coordinate to use for the splitting:
                longitude ('x') or latitude ('y'). (string)
     - eps -> degrees of which the training/validation sets are resized.
              Measure unit = degrees. (float)
     - tol -> percentage of tollerance of the size of the resultin training set
              with respect to the size it should have in a classical k-fold CV. (float)
    OUTPUT:
     - cleft, cright -> New coordinate bounds of the enlarged window. (float)
    """
    mask = (getattr(data.geometry, coord) < cleft) | (getattr(data.geometry, coord) > cright)
    window_size = data[mask].shape[0]
    right = True
    while window_size < (1-tol)*default_window_size:
        # Choose the direction (left or right) and modify the bounds
        if right: cright -= eps
        else: cleft += eps
        right = not right
        # Count number of points in the new Outer Window
        mask = (getattr(data.geometry, coord) < cleft) | (getattr(data.geometry, coord) > cright)
        window_size = data[mask].shape[0]

    return cleft, cright

def ReduceOuterWindow(data, default_window_size, cleft, cright, cmin, cmax,
                      coord, eps, tol, right = True):
    """
    INPUT:
     - data -> GeoDataFrame with geometry column.
     - default_window_size -> desired size of the window in terms of points contained. (int)
     - cleft, cright -> bounds of the considered coordinate. (float)
     - cmin, cmax -> absolute minimum and maximum values of the coordinate. (float)
     - coord -> coordinate to use for the splitting:
                longitude ('x') or latitude ('y'). (string)
     - eps -> degrees of which the training/validation sets are resized.
              Measure unit = degrees. (float)
     - tol -> percentage of tollerance of the size of the resultin training set
              with respect to the size it should have in a classical k-fold CV. (float)
     - right -> boolean variable which allows to start resizing on the right side of
                the window. (bool)
    OUTPUT:
     - cleft, cright -> New coordinate bounds of the enlarged window. (float)
    """
    mask = (getattr(data.geometry, coord) < cleft) | (getattr(data.geometry, coord) > cright)
    window_size = data[mask].shape[0]
    while window_size > (1+tol)*default_window_size:
        # Choose the direction (left or right) and modify the bounds
        if right: cright += eps
        else: cleft -= eps

        if coord == 'x':
            ## The following instructions are needed only in case we are dealing
            ## with longitude coordinates, since the projection of the spherial
            ## surface of the Earth on the plane, hides the continuity of the left
            ## border with the right border, and vice versa.
            right = not right

            # Check whether we cross the left and right bound
            if cleft < cmin: cleft = cmax - abs(cleft - cmin)
            if cright > cmax: cright = cmin + abs(cright - cmax)

            if cleft > cright:
                mask = (getattr(data.geometry, coord) < cleft) & (getattr(data.geometry, coord) > cright)
            else:
                mask = (getattr(data.geometry, coord) < cleft) | (getattr(data.geometry, coord) > cright)
        else:
            ## Here we are dealing with latitude coordinates, that are not affected by
            ## the same ambiguity of longitude coordinates, due to the projection.
            mask = (getattr(data.geometry, coord) < cleft) | (getattr(data.geometry, coord) > cright)

        window_size = data[mask].shape[0]
    return cleft, cright

def SpatialCVSplit(data, k = 5, coord = 'x', buffer = 1, eps = 1, tol = 0.01):
    """
    INPUT:
     - data -> GeoDataFrame with at least the geometry column.
     - k -> number of folds of the cross-validation. (int)
     - coord -> coordinate to use for the splitting: longitude ('x') or latitude ('y'). (string)
     - buffer -> size of the buffering area between the validation and the training set for each
                 fold (area not included neither in training nor in validation).
                 Measure unit = degrees ((1° ~ 111.319 km at the equator)
     - eps -> degrees of coordinate of which the training/validation sets are resized each iteration.
              Measure unit = degrees. (float)
     - tol -> percentage of tollerance of the size of the resulting training set with respect to
              the size it should have in a classical k-fold CV. (float)
    OUTPUT:
     - list(zip(train_ids, val_ids)) -> list of touples of the form (train, val).
              Both, train and val, are numpy arrays containing the (pandas) indices of
              the points that are contained in each set.
              Each touple represent a split of the data for the CV.
    """
    ### Outer Window = Training set of the considered fold

    # Minimum coordinate
    cmin = getattr(data.geometry, coord).min()
    # Maximum coordinate
    cmax = getattr(data.geometry, coord).max()
    # Bounds of k equally large intervals
    c = np.linspace(start = cmin, stop = cmax, num = k+1)
    # Training size of a classic CV split
    default_train_size = data.shape[0] - data.shape[0]/k

    # Create the coordinate bounds for the training set
    ## In case of latitude coordinates, left and right should be interpreted
    ## respectively, as bottom and top
    tbl = [] ## Left bounds
    tbr = [] ## Right bounds
    for i in range(0, k):
        print(f"Preparing training set {i+1} bounds")
        # Get default coordinate bounds for training and validation set
        cleft, cright = c[i:i+2]
        # Measure the training size
        mask = (getattr(data.geometry, coord) < cleft) | (getattr(data.geometry, coord) > cright)
        train_size = data[mask].shape[0]
        # Resize the training when it is too small...
        if train_size < (1-tol)*default_train_size:
            cleft, cright = EnlargeOuterWindow(data, default_train_size, cleft, cright,
                                               coord, eps, tol)
        # ... or too large with respect to the classic size default_train_size
        elif train_size > (1+tol)*default_train_size:
            if i <= k-1:
                cleft, cright = ReduceOuterWindow(data, default_train_size, cleft, cright, cmin, cmax,
                                                  coord, eps, tol)
            else:
                ## In case of latitude, the Outer Window of the last ("northest") fold cannot be
                ## enlarged on the right (top) side, that is above the Artic.
                ## This is the case treated by this exception
                cleft, cright = ReduceOuterWindow(data, default_train_size, train_size,
                                                  cleft, cright, cmin, cmax, coord, eps, tol, right = False)

        # Save the new coordinate bounds for the training set
        tbl.append(cleft)
        tbr.append(cright)

    print("\nPreparing validation sets bounds")
    # Create the coordinate bounds for the validation set
    # A buffering area is inserted between the validation and the training set
    vbl = [c+buffer if c+buffer<cmax else cmin+abs(c+buffer-cmax) for c in tbl] ## Left Bounds
    vbr = [c-buffer if c-buffer>cmin else cmax-abs(c-buffer-cmin) for c in tbr] ## Right Bounds

    print("\nAssigning points to training and validation sets\n")
    # Extract pandas indices of points in training and validation set for each fold.
    val_ids = []
    train_ids = []
    for i in range(0,k):
        # Left bound larger than right bound
        if tbl[i] > tbr[i]:
            train_mask = (getattr(data.geometry, coord) > tbr[i]) & (getattr(data.geometry, coord) < tbl[i])
            val_mask = (getattr(data.geometry, coord) <= vbr[i]) | (getattr(data.geometry, coord) >= vbl[i])
        # Left bound smaller than right bound
        else:
            train_mask = (getattr(data.geometry, coord) < tbl[i]) | (getattr(data.geometry, coord) > tbr[i])
            val_mask = (getattr(data.geometry, coord) >= vbl[i]) & (getattr(data.geometry, coord) <= vbr[i])
        train_ids.append(data[train_mask].index.values)
        val_ids.append(data[val_mask].index.values)

    return list(zip(train_ids, val_ids))

def FoldsStats(folds_indices, tot_size):
    """
    INPUT:
     - folds_indices -> list of touples of the form (train, val) for CV.
              Both, train and val, are numpy arrays containing the (pandas) indices of
              the points that are contained in each set.
     - tot_size -> total size of the dataset in terms of number of points.
    OUTPUT:
     - FOLDS_STATS -> DataFrame containing counts and proportions of training and
                      validation sets and buffering areas.
    """
    ts, vs = zip(*[(len(train),len(val)) for train, val in folds_indices])
    k = len(folds_indices)

    FOLDS_STATS = pd.DataFrame({'TrainSetSize': ts,
                                'ValidationSetSize': vs,
                                'BufferingSize': [tot_size-(t+v) for t,v in zip(ts,vs)],
                                'TrainSetProp': np.round([t*100/tot_size for t in ts], 2),
                                'ValidationSetProp': np.round([v*100/tot_size for v in vs], 2),
                                'BufferingProp': np.round([100-(t+v)*100/tot_size for t,v in zip(ts,vs)],2)},
                               index = [f'Fold{i}' for i in range(1,k+1)])
    return FOLDS_STATS


def Index2RowNum(tv_ids, data):
    """
    INPUT:
     - tv_ids -> list of touples of the form (t, v). Both t and v
                 are arrays of pandas indices of elements contained in data.
     - data -> DataFrame.

    OUTPUT:
     - A list of touples of the form (t, v). Both t and v are lists
       of the row numbers corresponding to the rows where the elements
       indexed in tv_ids are contained in data .
    """

    print('Converting pandas indices into row numbers.')
    val_ids = []
    train_ids = []
    for train, val in tv_ids:
        # Find location (row number) in data of each element in train and val
        T = [data.index.get_loc(t) for t in train]
        V = [data.index.get_loc(v) for v in val]
        # Append the lists of row numbers
        train_ids.append(T)
        val_ids.append(V)

    return list(zip(train_ids, val_ids))

def Whiskers(data):
    """
    INPUT:
     - data -> data values. (series)

     OUTPUT:
      - min_w, max_w -> minimum/maximum whisker of the input data. (float)
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    IQR = q3-q1

    max_w = q3 + 1.5*IQR
    min_w = q1 - 1.5*IQR

    if min_w < data.min(): min_w = data.min()
    if max_w > data.max(): max_w = data.max()

    return min_w, max_w


def GeoPlot(GDF, col, cmap='viridis', marker_size=13,
            boundary_color='white', land_color="whitesmoke", 
            centred=False, outliers=False, vmin=None, vmax=None, 
            valuesoncmap=[], valuesoncmap_color='yellow',
            show=True, save=False, plot_dir='Output/Plots', title='GeoPlot', 
            save_params={'format':'jpg'}, Antarctica=True):
    """
    INPUT:
      - GDF -> GeoDataFrame with geometry column and other data columns
      - col -> name of the column containing data to plot. (string)
      - cmap -> colormap name (string)
      - marker_size -> size of data points on the map. (int)
      - boundary_color -> countries' borders colour. (string)
      - land_color -> lands colour. (string)
      - centred -> If data are spread around zero, a diverging colormap will be 
                   centred on zero. (bool) It is useful in case of diverging 
                   measures aroung a reference value, like temperatures around 0
      - outliers -> Remove outliers from the range covered by the colormap.
                    They will be represented with the same colour of the minimum
                    or maximum value represented in the colormap. (bool) This is
                    useful when the presence of outliers makes the other values
                    indistinguishable by their colour in the plot.
      - vmin, vmax -> Minimum and maximum values considered in the colormap's
                      range. (int) Smaller or larger values are represented with
                      same colour of the minimum and maximum values respectively.
      - valuesoncmap -> Specific values that will be represented on the colormap
                        bar as straigth vertical lines. (float or list of floats)
      - valuesoncmap_color -> Colour of lines on the colormap bar. (string)                  
      - show -> Chose to show or not the plot. (bool) 
      - save -> Choose to save or not the plot. (bool) If 'show'=True, '
                save'=False and a file named after 'title' is already existing 
                in the 'plot_dir' directory, then this file is immediately 
                displayed avoiding other computations. If 'save'=True, the 
                already existing file is replaced by the one just created. 
      - plot_dir -> Saving directory for the plot. (path string)
      - title -> Name of the plot file (no file extension). (string)
      - Antarctica -> Choose to plot of not the Antartic continent. (bool)
    """
    # Directory to store the plot
    file_dir = os.path.join(plot_dir, f"{title}.{save_params['format']}")
    if show and os.path.exists(file_dir) and not save:
        display(Image(filename=file_dir, retina=True))
    else:
        # Import the geodataframe containing the map of the world
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        # Remove Antarctica if needed
        if not Antarctica:
            world = world[(world.name!="Antarctica")]
            GDF = RemoveAntarctica(GDF)

        # Remove outliers from the colormap and centre the colormap on 0 if needed
        if outliers:
            min_w, max_w = Whiskers(GDF[col])
            if centred:
                norm = TwoSlopeNorm(vmin=min_w, vcenter=0, vmax=max_w)
            else:
                norm = Normalize(vmin=min_w, vmax=max_w)
        else:
            if vmin is None: vmin = GDF[col].min()
            if vmax is None: vmax = GDF[col].max()
            if centred:
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)

        with sns.axes_style("white"):
            fig, ax = plt.subplots(figsize=(50,27))

            # Legend adjustments
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="5%", pad=0.1)
            cax.tick_params(labelsize=40)

            # Plot of the data points
            GDF.plot(column=col, zorder=2, ax=ax, cmap=cmap, norm=norm, cax=cax,
                     legend=True, legend_kwds={'orientation': 'horizontal'},
                     markersize=marker_size, marker='h')

            # Plot of the world on the backgroud
            world.plot(ax=ax, facecolor=land_color, edgecolor="none", zorder=1)\
                 .set_facecolor('#adcfeb')#('#A8C5DD')
            # Plot of the countries' borders on the foreground
            world.boundary.plot(ax=ax, color=boundary_color, zorder=3)

            # Hide axis
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            #####
            for v in list(valuesoncmap):
                cax.vlines(v, -1, 1, colors=valuesoncmap_color, linewidth=5)
            
            fig.tight_layout()
            if save:
                plt.savefig(file_dir, **save_params)
            if show:
                plt.show()
            else:
                plt.close()
                

def EmpiricalDistribution(data, col, check_na = True,
                          save = False, plot_dir = 'Output/Plots', title = ''):
    """
    INPUT:
      - data -> (Geo)DataFrame with multiple data columns
      - col -> name of the column containing data to analyse. (string)
      - save -> Choose to save or not the plots. (bool)
      - plot_dir -> Saving directory for the plots. (path string)
      - title -> Name of the plot files. (string)
    """
    # Remove NA values if needed
    if check_na:
        BV = data.loc[data[col]>-100, col].copy()
    else: BV = data[col].copy()

    d = data.shape[0] - BV.shape[0]
    if d>0:
        print(f'This feature has {d} not available values.')

    fig, ax = plt.subplots(ncols = 2, figsize = (16,4))

    # Frequency histogram
    sns.distplot(BV, hist = True, kde = False, ax = ax[0], axlabel = False )
    ax[0].set_ylabel('Frequency')

    # Density histogram
    sns.distplot(BV, hist = True, kde = True, color = 'darkblue',
                 ax = ax[1], axlabel = False )
    ax[1].set_ylabel('Density')

    fig.tight_layout()
    if save:
        plt.savefig(os.path.join(plot_dir, f'{title}_EmpDist.jpg'), optimize = True,
                    box_inches = 'tight', dpi = 200)

    fig, ax = plt.subplots(figsize = (16,4))

    # Boxplot
    sns.boxplot(BV, ax = ax, color = 'skyblue', fliersize=3)
    ax.set_xlabel('')

    fig.tight_layout()
    if save:
        plt.savefig(os.path.join(plot_dir, f'{title}_Boxplot.jpg'), optimize = True,
                    box_inches = 'tight', dpi = 200)

    plt.show()

#def EmpiricalLogDistribution(data, bv_idx, eps = 1e-5):
#    D = data.copy()
#    if D[f'BIO{bv_idx}_Centroid'].any(0):
#        D[f'BIO{bv_idx}_Centroid'] += eps
#    D[f'BIO{bv_idx}_Centroid'] = np.log(D[f'BIO{bv_idx}_Centroid'])
#    EmpiricalDistribution(D, bv_idx)

def RemoveAntarctica(data):
    return data[data.geometry.y>-60].copy()
    
            
def VisualiseTree(dtr, md, fn=[f[:5] for f in bv_cols], show=True, save=False,
                  plot_dir='Output/Plots', title='DTR', plot_format='png'):
    """
    INPUT:
     - dtr -> regression tree. (tree object)
     - md -> maximum depth to be visualised. (int)
     - fn -> names of the features as represented in the tree. (list of strings)
     - show -> Choose to display or not the plot. (bool)
     - save -> Choose to save or not the plots. (bool)
     - plot_dir -> Saving directory for the plots. (path string)
     - title -> Name of the plot files (No file extension). (string)
    """
    file_dir = os.path.join(plot_dir, f"{title}.{plot_format}")
    if show and os.path.exists(file_dir) and not save:
        display(Image(filename=file_dir))
    else:
        # Create an empty object to receive a string buffer in DOT format.
        dot_data = StringIO()

        # Exports the tree in DOT format into the out_file
        export_graphviz(dtr, out_file=dot_data, max_depth=md, feature_names=fn,
                        proportion=True, filled=True, rounded=True)

        # Use the DOT object to create the graph.
        tree = graph_from_dot_data(dot_data.getvalue())

        if save: tree.write(file_dir, format=plot_format)

        if show: display(Image(tree.create(format=plot_format)))


def FeatureImportanceRanking(tree, feature_labels, show=True, save=False,
                             plot_dir='Output/Plots', title='FIR', 
                             save_params={'format':'jpg'}):
    
    fig, ax = plt.subplots(figsize=(10, 7))

    FI = pd.Series(tree.feature_importances_, index=feature_labels).sort_values()

    FI.plot(kind='barh', color='darkcyan', legend=False,
            fontsize=15, width=0.9, ax=ax)
    ax.get_xaxis().set_visible(False)
    ax.set(facecolor='white')
    for i, imp in enumerate(FI):
        ax.text(imp+0.001, i-0.13, str(round(imp, 3)))

    fig.tight_layout()

    if save:
        plt.savefig(os.path.join(plot_dir, f"{title}.{save_params['format']}"),
                    **save_params)
    if show:
        plt.suptitle("Feature Importance Ranking of selected DTR", y=1.1, size=20)
        plt.show()
    else:
        plt.close()
            
            
def OneSeRule(GS, parameter, cond, cv, param_grid = {},
              mean_error_col='mean_test_score', std_error_col='std_test_score'):
    parameter = f'param_{parameter}'
    if param_grid:
        mask = eval(" & ".join([f"(GS['param_{p}'] == {param_grid[p]})" for p in param_grid.keys()]))
        R = GS[mask].copy()
    else:
        R = GS.copy()
    
    best_id = R[mean_error_col].idxmin()
    best_error = R.loc[[best_id], mean_error_col].item()
    best_param = R.loc[[best_id], parameter].item()
    best_se = R.loc[[best_id], std_error_col].item()/np.sqrt(cv)
    
    if cond == 'min':
        mask = eval(" & ".join([f"(R['{parameter}'] < best_param)",
                                f"(R['{mean_error_col}'] <= best_error+best_se)"]))
    else:
        mask = eval(" & ".join([f"(R['{parameter}'] > best_param)",
                                f"(R['{mean_error_col}'] <= best_error+best_se)"]))
    R_SE = R[mask]
    
    if R_SE.empty:
        print("No 1-standard-error complexity parameter can be obtained."
              "The best parameter has been returned instead.")
        return {'alpha':best_param, 'error': best_error}
    else:
        mask = eval(f"(R_SE['{parameter}'] == R_SE['{parameter}'].{cond}())")
        best_1se_param = R_SE[mask][parameter].item()
        best_1se_error = R_SE[mask][mean_error_col].item()  

        return {'alpha':best_1se_param, 'error':best_1se_error}
    

def ConvertScores(GS_OUT, cv, convertion):

    df = GS_OUT.copy()
    for key, (new_key, t) in convertion.items():
        df.rename(columns={f'split{f}_test_{key}':
                           f'split{f}_test_{new_key}' for f in range(cv)},
                  inplace=True)
        df.rename(columns={f'mean_test_{key}': f'mean_test_{new_key}',
                           f'std_test_{key}': f'std_test_{new_key}',
                           f'rank_test_{key}': f'rank_test_{new_key}'},
                  inplace=True)
        
        df[[f'split{f}_test_{new_key}' for f in range(cv)]] = \
        df[[f'split{f}_test_{new_key}' for f in range(cv)]].apply(t)
        df[f'mean_test_{new_key}'] = df[f'mean_test_{new_key}'].apply(t)

    return df



def GridSearchWarmStart(X, y, estimator, param_grid,
                        scoring={'R2E': lambda y_true, y_pred: 1-r2(y_true=y_true, y_pred=y_pred)},
                        cv=10, oob_score=True):
    # List of the explored hyperparameters names. 
    param_cols = list(param_grid.keys())

    # Number of n_estimators values explored
    if 'n_estimators' in param_grid:
        N = len(param_grid['n_estimators'])
        # Move n_estimators at the end of the list (needed for sorting df values 
        # in the next steps) 
        param_cols.append(param_cols.pop(param_cols.index('n_estimators')))
        # Count all combinations of hyperparameters values excluding n_estimators
        if len(param_cols[:-1]) == 0: # If only n_estimators are explored 
            count = 1   
        else: # If additional hyperparameters are explored
            count = np.prod([len(param_grid[c]) for c in param_cols[:-1]]) 
    else: 
        N = 1
        count = np.prod([len(param_grid[c]) for c in param_cols])
        
    # Dataframe that will contain hyperparameter combinations and 
    # model performances (OOB_score) in the grid search
    df = pd.DataFrame(list(ParameterGrid(param_grid)))
    df.sort_values(by=param_cols, inplace=True, ignore_index=True)
    
    if oob_score:
        print(f"Grid Search with Out-Of-Bag samples.\n"
              f"Fitting {df.shape[0]} candidates.\n")
        # Column that will contain the OOB score
        df['OOB_R2E'] = np.nan 
        
        # Grid-Search
        # Cycle on every combination of hyperparameter values, excluding n_estimators
        for i in range(count):
            # Base estimator
            rgr = clone(estimator)
            rgr.set_params(oob_score=oob_score)
            for i_n in range(N):
                params = {p: df.loc[i*N+i_n, p] for p in param_cols}
                # Update n_estimators and set the other parameters.
                # For every i, all parameter but n_estimators are fixed,
                # so here we set the parameters when i_n=0, but for i_n>0
                # we update n_estimators only
                rgr.set_params(**params)
                # Fitting the Random Forest
                rgr.fit(X, y)
                # OOB error
                df.at[i*N+i_n,'OOB_R2E'] = 1-rgr.oob_score_
            print(f"   {(i+1)*N} candidates explored.")
        print()
        
    if pd.notna(cv):
        print(f"Grid Search with k-fold Cross-Validation.\n"
              f"Fitting {cv} folds for each of {df.shape[0]} candidates, totalling {cv*df.shape[0]} fits.\n")
    
        # Additional dataframe columns that will be filled during GS
        cols = []
        for m in scoring.keys():
            cols.extend([f'split{i}_test_{m}' for i in range(cv)])
            cols.extend([f'{l}_test_{m}' for l in ['mean','std']])
        df[cols] = np.nan 
        
        # Splitting the training set into cv folds 
        folds = [f.sample(frac=1) for f in np.array_split(X, cv)]
        
        # Grid-Search
        for f in range(cv): # Cycle on each fold
            # Train set: Everything but the selected fold
            train = folds.copy()
            del train[f]
            X_train = pd.concat(train, sort=False)
            y_train = y.loc[X_train.index]
            # Test set: Selected fold
            Xtest = folds[f]
            ytest = y.loc[Xtest.index]
            
            # Cycle on every combination of hyperparameter values, 
            # excluding n_estimators
            for i in range(count):
                # Base Random Forest
                rgr = clone(estimator)
                for i_n in range(N):
                    params = {p: df.loc[i*N+i_n, p] for p in param_cols}
                    # Update n_estimators and set the other parameters.
                    # For every i, all parameter but n_estimators are fixed,
                    # so here we set the parameters when i_n=0, but for i_n>0
                    # we update n_estimators only
                    rgr.set_params(**params)
                    # Fitting the Random Forest
                    rgr.fit(X_train, y_train)
                    
                    ypred = rgr.predict(Xtest)
                    for m in scoring.keys():
                        df.at[i*N+i_n, f'split{f}_test_{m}'] = scoring[m](y_true=ytest, y_pred=ypred)
            print(f"Fold {f+1}: {df.shape[0]} fits done.")
        
        for m in scoring.keys():
            cols = [f'split{i}_test_{m}' for i in range(cv)]
            df[f'mean_test_{m}'] = df[cols].mean(axis=1)
            df[f'std_test_{m}'] = df[cols].std(axis=1)
        
    # Rename parameters columns like in sklearn.model_selection.GridSearchCV
    df.rename(columns = {key: f'param_{key}' for key in param_grid.keys()},
            inplace=True) 
    
    print("Grid Search performed. \n")
    
    return df
