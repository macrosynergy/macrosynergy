"""example/macrosynergy/panel/make_blacklist.py"""


cids = ['AUD', 'GBP', 'CAD', 'USD']
cols = ['earliest', 'latest', 'mean_add', 'sd_mult']
df_cid1 = pd.DataFrame(index=cids, columns=cols)

df_cid1.loc['AUD'] = ['2010-01-01', '2020-12-31', 0, 1]
df_cid1.loc['GBP'] = ['2011-01-01', '2020-11-30', 0, 1]
df_cid1.loc['CAD'] = ['2011-01-01', '2021-11-30', 0, 1]
df_cid1.loc['USD'] = ['2011-01-01', '2020-12-30', 0, 1]

cols = ['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef']
df_xcat1 = pd.DataFrame(index=['FXXR_NSA', 'FXCRY_NSA'],  columns=cols)
df_xcat1.loc['FXXR_NSA'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
df_xcat1.loc['FXCRY_NSA'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
df1 = make_qdf(df_cid1, df_xcat1, back_ar=0.05)

df_xcat2 = pd.DataFrame(index=['FXNONTRADE_NSA'],  columns=['earliest', 'latest'])
df_xcat2.loc['FXNONTRADE_NSA'] = ['2010-01-01', '2021-11-30']
black = {'AUD': ('2010-01-12', '2010-06-14'),
         'USD': ('2011-08-17', '2011-09-20'),
         'CAD_1': ('2011-01-04', '2011-01-23'),
         'CAD_2': ('2013-01-09', '2013-04-10'),
         'CAD_3': ('2015-01-12', '2015-03-12'),
         'CAD_4': ('2021-11-01', '2021-11-20')}

print(black)
df2 = make_qdf_black(df_cid1, df_xcat2, blackout=black)

df = pd.concat([df1, df2]).reset_index()

dates_dict = make_blacklist(df, xcat='FXNONTRADE_NSA', cids=None,
                            start=None, end=None)

# If the output, from the below printed dictionary, differs from the above defined
# dictionary, it should be by a date or two, as the construction of the dataframe,
# using make_qdf_black(), will account for the dates received, in the dictionary,
# being weekends. Therefore, if any of the dates, for the start or end of the
# blackout period are Saturday or Sunday, the date for will be shifted to the
# following Monday. Hence, a break in alignment from "blackout" to "dates_dict".
print(dates_dict)