bv_cols = ['BIO01_Centroid','BIO02_Centroid','BIO03_Centroid','BIO04_Centroid',
           'BIO05_Centroid','BIO06_Centroid','BIO07_Centroid','BIO08_Centroid',
           'BIO09_Centroid','BIO10_Centroid','BIO11_Centroid','BIO12_Centroid',
           'BIO13_Centroid','BIO14_Centroid','BIO15_Centroid','BIO16_Centroid',
           'BIO17_Centroid','BIO18_Centroid','BIO19_Centroid']

bv_ids = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
          '11', '12', '13', '14', '15', '16', '17', '18', '19']

bv_names = ['annual mean temperature', 'mean diurnal range', 'isothermality',
            'temperature seasonality', 'max temperature of warmest month',
            'min temperature of coldest month', 'annual temperature range',
            'mean temperature of wettest quarter',
            'mean temperature of driest quarter',
            'mean temperature of warmest quarter',
            'mean temperature of coldest quarter', 'annual precipitations',
            'precipitations of wettest month', 'precipitations of driest month',
            'precipitation seasonality', 'precipitations of wettest quarter',
            'precipitations of driest quarter',
            'precipitations of warmest quarter',
            'precipitations of coldest quarter']

id2bv = dict(zip(bv_ids, bv_names))

bv_labels = [n+' '+i for i, n in zip(bv_ids, bv_names)]