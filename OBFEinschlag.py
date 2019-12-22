################################################################################
#
#   fuc_tbl_hiebsatzbilanz (to, filterx)
#
################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

class OBFEinschlag(object):

    def __init__(self, data):

        self.data = data

        # fill nan with 0
        self.data = self.data.fillna(0)


    def printit(self):
        print(self.data)
        return(self.data)


    #############################
    ### filter data
    #############################
    def filter_data(self, fr, group=True):

        if (fr[0] != 0) & (fr[0]<1000):
            data_filter = self.data[self.data['Forstrevier'].isin(fr)]
        elif (fr[0] != 0) & (fr[0]>=1000):
            data_filter = self.data[self.data['Teiloperats-ID'].isin(fr)]
        else:
            data_filter = self.data

        es = data_filter[data_filter['Hiebssatz Kennzahlty'] == 'JES']
        hs = data_filter[data_filter['Hiebssatz Kennzahlty'] == 'JHS']
        bilz = 0
        bilz_ub = data_filter[data_filter['Hiebssatz Kennzahlty'] == 'BJHSNU']

        if group:
            es = es.groupby('Jahr').sum()*(-1)
            hs = hs.groupby('Jahr').sum()
            bilz = (es-hs).cumsum()*(-1)
            bilz_ub = bilz_ub.groupby('Jahr').sum()

        return(es, hs, bilz, bilz_ub)


    #############################
    ### prepare data
    #############################
    def func_prepare_data(self, filterx, es, hs, bilz):

        if filterx == 'Ges':
            table = hs.loc[:,['VNLH']]
            table.columns = ['HS_LH']
            table['HS_LH'] = table['HS_LH'] + hs['ENLH']
            table['HS_NH'] = hs.loc[:,'VNNH'] + hs.loc[:,'ENNH']
            table['HS'] = table['HS_LH'] + table['HS_NH']

            # Einschlag
            table['ES_LH'] = es.loc[:,'VNLH'] + es.loc[:,'ENLH']
            table['ES_NH'] = es.loc[:,'VNNH'] + es.loc[:,'ENNH']
            table['ES'] = table['ES_LH'] + table['ES_NH']

            # bilanzierter Hiebsatz
            table['bil_LH'] = bilz.loc[:,'VNLH'] + bilz.loc[:,'ENLH']
            table['bil_NH'] = bilz.loc[:,'VNNH'] + bilz.loc[:,'ENNH']
            table['bil'] = table['bil_LH'] + table['bil_NH']
        else:
            # Hiebsatz
            table = hs.loc[:,[filterx + 'LH']]
            table.columns = ['HS_LH']
            table['HS_NH'] = hs.loc[:,[filterx + 'NH']]
            table['HS'] = table['HS_LH'] + table['HS_NH']

            # Einschlag
            table['ES_LH'] = es.loc[:,[filterx + 'LH']]
            table['ES_NH'] = es.loc[:,[filterx + 'NH']]
            table['ES'] = table['ES_LH'] + table['ES_NH']

            # bilanzierter Hiebsatz
            table['bil_LH'] = bilz.loc[:,[filterx + 'LH']]
            table['bil_NH'] = bilz.loc[:,[filterx + 'NH']]
            table['bil'] = table['bil_LH'] + table['bil_NH']

        return(table)


    #############################
    # get min & max
    #############################
    def max_min(self, fr):

        y_max = 0
        y_min = 0
        print(fr)
        es, hs, bilz, bilz_ub = self.filter_data(fr)
        for i in ['Ges', 'EN', 'VN']:
            table = self.func_prepare_data(i, es, hs, bilz)

            max = table.max().max()
            min = table.min().min()

            if max > y_max:
                y_max = max
            if min < y_min:
                y_min = min

            #self.y_max = y_max
            #self.y_min = y_min
        return(y_max, y_min)


    #############################
    ### jährlicher HS, mittlerer ES
    #############################
    def fuc_tbl(self, fr, kind='all'):

        table = []
        es, hs, bilz, bilz_ub = self.filter_data(fr)

        if kind == 'all':
            # einschlag
            table.append([es['Summe'].sum()/es['Summe'].shape[0],
            (es['ENLH'].sum()+es['ENNH'].sum())/es['Summe'].shape[0],
            (es['VNLH'].sum()+es['VNNH'].sum())/es['Summe'].shape[0]])

            # hiebsatz
            table.append([hs['Summe'].sum()/hs['Summe'].shape[0],
            (hs['ENLH'].sum()+hs['ENNH'].sum())/hs['Summe'].shape[0],
            (hs['VNLH'].sum()+hs['VNNH'].sum())/hs['Summe'].shape[0]])

        else:
            # einschlag
            table.append([(es['ENLH'].sum() + es['VNLH'].sum()) / es['Summe'].shape[0],
            (es['ENNH'].sum() + es['VNNH'].sum()) / es['Summe'].shape[0],
            es['ENLH'].sum() / es['Summe'].shape[0],
            es['ENNH'].sum() / es['Summe'].shape[0],
            es['VNLH'].sum() / es['Summe'].shape[0],
            es['VNNH'].sum() / es['Summe'].shape[0]])

            # hiebsatz
            table.append([(hs['ENLH'].sum() + hs['VNLH'].sum()) / hs['Summe'].shape[0],
            (hs['ENNH'].sum() + hs['VNNH'].sum()) / hs['Summe'].shape[0],
            hs['ENLH'].sum() / hs['Summe'].shape[0],
            hs['ENNH'].sum() / hs['Summe'].shape[0],
            hs['VNLH'].sum() / hs['Summe'].shape[0],
            hs['VNNH'].sum() / hs['Summe'].shape[0]])

        table.append([bilz_ub['Summe'].values[0],
        bilz_ub['ENLH'].values[0] + bilz_ub['ENNH'].values[0],
        bilz_ub['VNLH'].values[0] + bilz_ub['VNNH'].values[0]])

        return(table)


    # get all important time variables
    def get_time(self, es, hs):

        laufzeit = es['LfztJahre'].iloc[0]
        year_start = es['Jahr'].unique().min()
        year_now = es['Jahr'].unique().max()
        rest_laufzeit = laufzeit - (year_now +1 - year_start)

        hs_year = hs[hs['Jahr']==year_now].sum()[['ENLH', 'ENNH', 'VNLH', 'VNNH', 'Summe']]
        return(laufzeit, rest_laufzeit, year_start, year_now, hs_year)


    # calculate bilanzierter HS negativ berücksichtigt
    def bilanzierter_HS(self, es, hs):

        laufzeit, rest_laufzeit, year_start, year_now, hs_year = self.get_time(es, hs)

        hs_ges = hs_year * laufzeit
        es_sum = es[['ENLH', 'ENNH', 'VNLH', 'VNNH', 'Summe']].sum()
        hs_sum = hs[['ENLH', 'ENNH', 'VNLH', 'VNNH', 'Summe']].sum()
        bilz_hs = (hs_ges + es_sum) / rest_laufzeit

        if (bilz_hs['Summe']==float('inf')) or (bilz_hs['Summe']==float('-inf')):
            bilz_hs.iloc[:]=0

        return(bilz_hs)


    #############################
    ### pie
    #############################

    def fuc_tbl_pie(self, fr, hs_all):
        '''
            self.data = input self.data (dataframe)
            to = Teiloperat (int)
            filterx = Vornutzung / Endnutzung ('EN'|'VN')
        '''
        es, hs, bilz, bilz_ub = self.filter_data(fr, group=False)
        hses = []

        for i in ['Ges','EN','VN']:

            es_s = 0
            hs_s = 0

            if i == 'Ges':
                es_s = es['Summe'].sum()*-1
                hs_s = hs_all['Summe']
                self.dounat_chart(es_s, hs_s, 0)
                hses.append([es_s,hs_s])

            elif i == 'EN':
                es_s = (es['ENLH'].sum() + es['ENNH'].sum())*-1
                hs_s = hs_all['ENLH'] + hs_all['ENNH']
                self.dounat_chart(es_s, hs_s, 1)
                hses.append([es_s,hs_s])

            elif i == 'VN':
                es_s = (es['VNLH'].sum() + es['VNNH'].sum())*-1
                hs_s = hs_all['VNLH'] + hs_all['VNNH']
                self.dounat_chart(es_s, hs_s, 2)
                hses.append([es_s,hs_s])

        return(hses)


    def fuc_tbl_pie_lh_nh(self, fr, hs_all):

        es, hs, bilz, bilz_ub = self.filter_data(fr, group=False)
        hses = []

        for i in ['Ges','EN','VN']:

            es_s = 0
            hs_s = 0

            if i == 'Ges':
                es_lh = (es['ENLH'].sum() + es['VNLH'].sum())*-1
                hs_lh = (hs_all['ENLH'] + hs_all['VNLH'])
                es_nh = (es['ENNH'].sum() + es['VNNH'].sum())*-1
                hs_nh = (hs_all['ENNH'] + hs_all['VNNH'])
                self.dounat_chart_lh_nh(es_lh, hs_lh, es_nh, hs_nh, 0)
                hses.append([es_lh, hs_lh, es_nh, hs_nh])

            elif i == 'EN':
                es_lh = (es['ENLH'].sum())*-1
                hs_lh = (hs_all['ENLH'])
                es_nh = (es['ENNH'].sum())*-1
                hs_nh = (hs_all['ENNH'])
                self.dounat_chart_lh_nh(es_lh, hs_lh, es_nh, hs_nh, 1)
                hses.append([es_lh, hs_lh, es_nh, hs_nh])

            elif i == 'VN':
                es_lh = (es['VNLH'].sum())*-1
                hs_lh = (hs_all['VNLH'])
                es_nh = (es['VNNH'].sum())*-1
                hs_nh = (hs_all['VNNH'])
                self.dounat_chart_lh_nh(es_lh, hs_lh, es_nh, hs_nh, 2)
                hses.append([es_lh, hs_lh, es_nh, hs_nh])

        return(hses)


    #############################
    #
    #############################

    def dounat_chart(self, es_s, hs_s, name):
        if hs_s - es_s < 0:
            diff = 0
        else:
            diff = hs_s - es_s
        fig, ax = plt.subplots()
        ax.pie([diff,es_s], colors=[(75/255, 174/255, 75/255), (250/255, 25/255, 25/255)], startangle=90) #labels=['frei', 'genutzt']

        my_cycle = plt.Circle((0,0),0.8,color='white')
        p = plt.gcf()
        p.gca().add_artist(my_cycle)

        plt.savefig('tempp_' + str(name) + '.png', bbox_inches='tight', dpi=300)


    def dounat_chart_lh_nh(self, es_lh, hs_lh, es_nh, hs_nh, name):

        if hs_lh - es_lh < 0:
            now_lh = 1
            diff_lh = 0
        else:
            now_lh = es_lh/hs_lh
            diff_lh = (hs_lh-es_lh)/hs_lh

        if hs_nh - es_nh < 0:
            now_nh = 1
            diff_nh = 0
        else:
            now_nh = es_nh/hs_nh
            diff_nh = (hs_nh-es_nh)/hs_nh

        fig, ax = plt.subplots()
        ax.pie([now_lh, diff_lh, diff_nh, now_nh], colors=[(72/255, 122/255, 193/255), 'white', 'white', (85/255, 140/255, 54/255)], startangle=90) # colors=['green', 'red'], labels=['frei', 'genutzt'],

        my_cycle = plt.Circle((0,0),0.8,color='white')
        p = plt.gcf()
        p.gca().add_artist(my_cycle)

        plt.savefig('temp_lh_nh_' + str(name) + '.png', bbox_inches='tight', dpi=300)


    # calculate percentage of HS and ES for plotting of plot_hs_es_percent()
    def calc_hs_es_percent(self, es, hs, bilz_hs):

        laufzeit, rest_laufzeit, year_start, year_now, hs_year = self.get_time(es, hs)

        # HS
        hs_garph = np.ones(laufzeit) * hs_year['Summe']
        # index
        index_graph = np.arange(year_start, year_start+laufzeit)
        # create HS dataframe with years as index
        hs_garph = pd.DataFrame(hs_garph, index=index_graph, columns=['gesamt HS'])

        # ES
        es_g = es.groupby('Jahr').sum()*(-1)
        es_garph = es_g.loc[:,['Summe']]
        es_garph.columns = ['gesamt ES']

        # concatonate HS & ES
        all_graph = pd.concat([es_garph, hs_garph], axis=1)
        # fill nan with bilanz HS
        all_graph = all_graph.fillna(bilz_hs['Summe'])
        # compute percentage
        all_graph = all_graph/hs_garph.sum()[0] * 100
        # set all negativ values to 0
        all_graph.loc[all_graph['gesamt ES'] < 0, 'gesamt ES'] = 0

        return(all_graph)


    # create hs es percentage plot
    def plot_hs_es_percent(self, data, col):

        fig, ax = plt.subplots()
        data.T.plot(kind='barh', ax=ax, stacked=True, width=0.4, color=col, edgecolor = 'white', linewidth=2, figsize=(13,1.3), legend=False)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        ax.get_yaxis().set_visible(False)
        plt.savefig('temp_per.png', bbox_inches='tight', dpi=300)


    #############################
    ### 6.1.1 Einschlagsübersicht
    #############################

    def fuc_tbl_hiebsatzbilanz(self, fr, filterx, kind='all', y_max=0, y_min=0):
        '''
            self.data = input self.data (dataframe)
            to = Teiloperat (int)
            filterx = Vornutzung / Endnutzung ('EN'|'VN')
        '''
        es, hs, bilz, bilz_ub = self.filter_data(fr)

        ## make table
        table = self.func_prepare_data(filterx[0], es, hs, bilz)

        ## plot figure

        # prepare for plot ALL
        tableX = table.loc[:,['ES']]
        tableX['hs'] = table.loc[:,['HS']]
        tableX['bilz'] = table.loc[:,['bil']]
        tableX.columns = ['Einschlag', 'Hiebsatz', 'Bilanz']

        # decide the location of the legend
        up_down = table.max().max() + table.min().min()

        # plot

        if kind == 'all':
            hight = 6
        else:
            hight = 7

        fig = plt.figure(figsize=(5,hight))
        ax = fig.add_subplot(111)
        #ax.set_title('Hiebsatzbilanz - ' + filterx[1])
        '''
        ax.plot(tableX.index, tableX['Einschlag'], label="Einschlag")
        ax.plot(tableX.index, tableX['Hiebsatz'], label="Hiebsatz")
        ax.plot(tableX.index, tableX['Bilanz'], label="Bilanz")
        ax.fill_between(tableX.index, 0, tableX['Bilanz'], facecolor="none", edgecolor='green', hatch='/', alpha=0.5)
        '''
        #ax.fill_between(nutzungX.index, 0, nutzungX['bilz'], color='green', edgecolor='green', hatch='/', alpha=0.25)
        #ax1.bar(range(1, 5), range(1, 5), color='none', edgecolor='red', hatch="/", lw=1., zorder = 0)
        if kind == 'all':
            ax = self.ax_plot_all(ax, tableX)
        else:
            ax = self.ax_plot_lh_nh(ax, table)

        if up_down < 0:
            ax.legend(loc='lower left', frameon=False)
        else:
            ax.legend(loc='upper left', frameon=False)
        #ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)

        #ax.spines['left'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position(('data',0))
        ax.spines['top'].set_color('none')
        #ax.spines['left'].set_color('none')
        ax.spines['left'].set_visible(False)
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        if y_max > 0:
            ax.set_ylim([y_min,y_max])
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set(xlabel='Jahr', ylabel='Einschlag [efm]')

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(8)
            label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.1 ))

        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        plt.savefig('tempx_' + str(filterx[2]) + '.png', bbox_inches='tight', dpi=300)
        # clear plot
        plt.clf()
        plt.close()


        # table

        table.reset_index(level=0, inplace=True)
        table = round(table,0)
        table = table.astype(int)
        table.columns = ['Jahr', 'LH', 'NH', 'Sum', 'LH', 'NH', 'Sum', 'LH', 'NH', 'Sum']

        return (table)


################################################################################
    ### 6.1.1 Einschlagsübersicht

    def ax_plot_all(self, ax, table):

        ax.plot(table.index, table['Einschlag'], label="Einschlag", color =(75/250, 174/255, 75/255))
        ax.plot(table.index, table['Hiebsatz'], label="Hiebsatz", color=(255/255, 25/255, 25/255))
        ax.plot(table.index, table['Bilanz'], label="Bilanz", color=(68/255, 136/255, 184/255))
        ax.fill_between(table.index, 0, table['Bilanz'], facecolor="none", edgecolor=(68/255, 136/255, 184/255), hatch='/', alpha=0.5)
        return(ax)

    def ax_plot_lh_nh(self, ax, table):

        ax.plot(table.index, table['ES_LH'], label="Einschlag LH", color=(72/255, 122/255, 193/255))
        ax.plot(table.index, table['HS_LH'], label="Hiebsatz LH", color=(15/255, 49/255, 95/255))
        ax.plot(table.index, table['bil_LH'], label="Bilanz LH", color=(76/255, 174/255, 235/255))
        ax.fill_between(table.index, 0, table['bil_LH'], facecolor="none", edgecolor=(76/255, 174/255, 235/255), hatch='/', alpha=0.5)

        ax.plot(table.index, table['ES_NH'], label="Einschlag NH", color=(85/255, 140/255, 54/255))
        ax.plot(table.index, table['HS_NH'], label="Hiebsatz NH", color=(36/255, 61/255, 22/255))
        ax.plot(table.index, table['bil_NH'], label="Bilanz NH", color=(156/255, 227/255, 114/255))
        ax.fill_between(table.index, 0, table['bil_NH'], facecolor="none", edgecolor=(156/255, 227/255, 114/255), hatch='\\', alpha=0.5)
        return(ax)


    # make colors for plot_hs_es_percent()
    def create_colors(self, laufzeit, rest_laufzeit):
        col = []
        for i in range(laufzeit-rest_laufzeit):
            col.append((250/255, 25/255, 25/255))
        for i in range(rest_laufzeit):
            col.append((75/255, 174/255, 75/255))
        return(col)
