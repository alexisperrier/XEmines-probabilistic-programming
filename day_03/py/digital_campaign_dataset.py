def flat_columns(c):
    replace_dict = {" ":"_", "(":"", ")":"","/":"","-":"_","?":"_" }
    c = c.lower()
    for k,v in replace_dict.items():
        c = c.replace(k,v)
    return c

file = '/Users/alexis/amcp/ECU/data/mar15/Chicago+DCM+Cost_AJ.csv'
odf = pd.read_csv(file).dropna()
odf.columns = [ flat_columns(c) for c in odf.columns  ]

gdf = odf.groupby(by = ['date','company']).sum().reset_index()
gdf = gdf[['date', 'company', 'impressions','clicks']]

gdf.rename(columns = {'company': 'geoloc'}, inplace = True)

def clean(company):
    company = company.replace( 'Company','').replace( 'Chicago','').strip()
    company = ' '.join( [ w for w in company.split(' ') if len(w) >1      ]  )
    return company

gdf['geoloc'] = gdf.geoloc.apply(lambda d : clean(d) )

gdf = gdf[gdf.impressions > 0]

gdf['impressions'] = gdf.impressions.apply(lambda d : int(d/5) - np.random.randint(5) )
gdf['clicks'] = gdf.clicks.apply(lambda d : d + np.random.randint(10))
# gdf['impressions'] = gdf.impressions.apply(lambda d : int(d/10) )

gdf['ctr'] = gdf.clicks / gdf.impressions

gdf.sort_values(by = ['date', 'geoloc'], inplace = True)
gdf.reset_index(drop = True, inplace = True)
gdf.to_csv('digital_campaign.csv', index = False)
