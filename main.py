import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import requests
import numpy as np
from numpy.linalg import lstsq
import math

ramp = lambda u: np.maximum( u, 0 )
step = lambda u: ( u > 0 ).astype(float)

#this is unused
def SegmentedLinearReg( X, Y, breakpoints ):
    nIterationMax = 10

    breakpoints = np.sort(np.array(breakpoints))

    dt = np.min(np.diff(X))
    ones = np.ones_like(X)

    for i in range(nIterationMax):
        # Linear regression:  solve A*p = Y
        Rk = [ramp(X - xk) for xk in breakpoints]
        Sk = [step(X - xk) for xk in breakpoints]
        A = np.array([ones, X] + Rk + Sk)
        p = lstsq(A.transpose(), Y, rcond=None)[0]

        # Parameters identification:
        a, b = p[0:2]
        ck = p[2:2 + len(breakpoints)]
        dk = p[2 + len(breakpoints):]

        # Estimation of the next break-points:
        newBreakpoints = breakpoints - dk / ck

        # Stop condition
        if np.max(np.abs(newBreakpoints - breakpoints)) < dt / 5:
            break

        breakpoints = newBreakpoints
    else:
        print('maximum iteration reached')

    # Compute the final segmented fit:
    Xsolution = np.insert(np.append(breakpoints, max(X)), 0, min(X))
    ones = np.ones_like(Xsolution)
    Rk = [c * ramp(Xsolution - x0) for x0, c in zip(breakpoints, ck)]

    Ysolution = a * ones + b * Xsolution + np.sum(Rk, axis=0)

    return Xsolution, Ysolution


def logifunc(x,A,x0,k,off):
    return A / (1 + np.exp(-k*(x-x0)))+off

url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
#'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
r = requests.get(url, allow_redirects=True)
open('data/time_series_19-covid-Confirmed.csv', 'wb').write(r.content)

mydateparser = lambda x: pd.datetime.strptime(x, "%Y %m %d %H:%M:%S")
conf_df = pd.read_csv("data/time_series_19-covid-Confirmed.csv")
conf_df['Country/Region'] = conf_df['Country/Region'].str.replace("Mainland China", "China")
print(conf_df)




melted = conf_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'])

print(melted)
melted.rename(columns={'variable':'strdate', 'value':'conf'}, inplace=True)
melted['conf'] = melted['conf'].astype(float)
melted['date'] = pd.to_datetime(melted.strdate)
melted.drop(columns=['strdate'], inplace=True)

melted.sort_values(['Country/Region', 'Province/State', 'date'], inplace=True)

melted.to_csv("data/melted.csv")


melted.set_index(['Country/Region', 'Province/State', 'date'], inplace=True)
melted.sort_index(inplace=True)
#melted['diffs'] = np.nan

countrydaysum = melted.groupby(['Country/Region', 'date'])['conf'].sum().reset_index(name='conf_c')
countrydaysum['country_diff'] = countrydaysum.groupby(['Country/Region'])['conf_c'].diff()

countrydaysum.to_csv("data/countrydaysum.csv")


def func_exp(x, a, b):
    return a * np.exp(b * (x))


def func_seg(x, a1, b1, a2, b2, a3, amp, x0, k):
    r = np.piecewise(x,
                        condlist=[
                            x < a1,
                            np.logical_and(x < a2, x >= a1),
                            np.logical_and(x < a3, x >= a2),
                            x >= a3,
                            np.isnan(x)
                        ],
                        funclist=[
                            lambda x: 0,
                            lambda x: (x - a1) * b1,
                            lambda x: (x - a2) * b2 + (a2 - a1) * b1,
                            lambda x: logifunc(x, amp, x0, k, 0),
                            #(x - a3) * b3 + (a3 - a2) * b2 + (a2 - a1) * b1,
                            lambda x: 4
                        ])
    return r

def expfit(set):
    #print(set)
    if max(set) < 15:
        return "Not enough data, too low"
    try:
        start = 0
        exp1 = 0
        exp2 = 1000
        setmax = np.max(set)
        for x in range(0, len(set)):
            if set.iloc[x] > 0 and start == 0:
                start = x

            if set.iloc[x] >= 20 and exp1 == 0:
                exp1 = x - start - 1

            if set.iloc[x] >= 200 and exp2 == 1000 and setmax > 400:
                exp2 = x - start - 1

        subset = np.log(set[start:]).values
        if len(subset) < 5:
            return "Not enough data, too few"
        b1 = subset[exp1] / exp1



        def func_cseg(x, b2, amp, x0, k):
            return func_seg(x, 0, b1, exp1, b2, exp2, amp, x0, k)

        sigmas = np.zeros((1, len(subset)))
        sigmas[:exp1] = 0.25
        sigmas[exp1:] = 1

        fit = curve_fit(func_cseg, np.arange(0, len(subset)), subset, p0=(0.5, 100, 1, 0.1)
                        #, sigma=np.ones(len(subset)) * 0.5
                        )
        if exp2 < 1000:
            initialBreakpoints = [exp1, exp2]
        else:
            initialBreakpoints = [exp1]
        # try:
        #     fit2 = SegmentedLinearReg(np.arange(0, len(subset)), subset, initialBreakpoints)
        # except np.linalg.LinAlgError:
        #     fit2 = ["Err", "Err"]

        b2 = fit[0][0]
        amp = fit[0][1]
        x0 = fit[0][2]
        k = fit[0][3]
        a2 = exp1
        a3 = exp2

        if len(subset) == 43:
            print("Italy?")

        doubling = 0

        if exp2 == 1000:
            doubling = math.log(2) / b2
            amp = 0
        maxx = len(set) - 1 - start
        model = func_seg(float(maxx), 0, b1, a2, b2, a3, amp, x0, k)
        cur_model = model
        days95 = 0

        if exp2 < 1000:
            double_model = np.log(np.exp(model) * 2)
            double_before = 0
            prev_model = model
            for x in range(maxx + 1, maxx + 150):
                model = func_seg(float(x), 0, b1, a2, b2, a3, amp, x0, k)
                if model > double_model:
                    double_before = x
                    break
                prev_model = model


            if double_before != 0:
                if len(subset) == 43:
                    print("Italy?")
                doubling = x - (maxx + 1)
                doubling += 1 - ((double_model - prev_model) / (model - prev_model))

            for x in range(maxx + 1, maxx + 300):
                model2 = func_seg(float(x), 0, b1, a2, b2, a3, amp, x0, k)
                if np.exp(model2) > np.exp(amp) * 0.95:
                    days95 = x
                    break

        if doubling == 0:
            amp = 0
        return [b1, b2, amp, x0, k, start, a2, a3, doubling,  np.exp(cur_model), len(subset), days95]
    except RuntimeError as err:
        print(err)
        return "Error: " + str(err) + "\n" + str(set)

fits = countrydaysum.groupby(['Country/Region']).agg({'conf_c': expfit})

fits_conf_c = fits.conf_c[fits.conf_c.apply(lambda x: not isinstance(x, str))]
fits['b1'] = fits_conf_c.apply(lambda x: x[0])
fits['b2'] = fits_conf_c.apply(lambda x: x[1])
fits['amp'] = fits_conf_c.apply(lambda x: x[2])
fits['x0'] = fits_conf_c.apply(lambda x: x[3])
fits['k'] = fits_conf_c.apply(lambda x: x[4])
fits['start'] = fits_conf_c.apply(lambda x: x[5])
fits['exp1_est'] = fits_conf_c.apply(lambda x: x[6])
fits['exp2_est'] = fits_conf_c.apply(lambda x: x[7])
fits['doubling'] = fits_conf_c.apply(lambda x: x[8])
fits['cur_model'] = fits_conf_c.apply(lambda x: x[9])
fits['valid_days'] = fits_conf_c.apply(lambda x: x[10])
fits['days95'] = fits_conf_c.apply(lambda x: x[11])
#fits.rename(columns={"conf_c", "fits"}, inplace=True)

# def resid(set):
#     #print(set)
#     if max(set) < 15:
#         return "Not enough data"
#     try:
#         fit = curve_fit(func_exp, range(0, len(set)), set, p0=(0.5, 0.2, 10))
#         return fit
#     except:
#         return "Error: " + str(set)
#
# fits = countrydaysum.groupby(['Country/Region']).agg({'conf_c': expfit})



countrydaysum.set_index(['Country/Region', 'date'])

fits.drop(columns=['conf_c'], inplace=True)



def make_slope(row):

#     if row['exp2_est'] < 1000:
#         return row['b3']

     return row['b2']


fits['slope'] = fits.apply(make_slope, axis=1)
fits.to_csv("data/fits.csv")

countrydaysum = countrydaysum.merge(fits, on=['Country/Region'], how='inner')


mindate = countrydaysum.date.min()

def row_model(group):
    if group['b1'] == np.nan:
        return None
    if math.isnan(group['b1']):
        return None
    x = (group['date'] - mindate).days - group['start']
    a1 = 0
    b1 = group['b1']
    b2 = group['b2']
    amp = group['amp']
    x0 = group['x0']
    k = group['k']
    a2 = group['exp1_est']
    a3 = group['exp2_est']
    def proj(x):
        return func_seg(x, 0, b1, a2, b2, a3, amp, x0, k)
    return np.exp(proj(x))



#def model(data):
#    return data.apply(row_model, axis=1)


countrydaysum['model'] = countrydaysum.apply(row_model, axis=1)
countrydaysum.loc[countrydaysum.model == 1, 'model'] = 0
countrydaysum.loc[countrydaysum['Country/Region'] == "China", 'model'] = countrydaysum.conf_c
countrydaysum.loc[countrydaysum['Country/Region'] == "China", 'err'] = 0


countrydaysum['err'] = np.log(countrydaysum['model']) - np.log(countrydaysum.conf_c) #np.log(countrydaysum['model']) - np.log(countrydaysum.conf_c)
countrydaysum.loc[countrydaysum.conf_c < 10, 'err'] = 0
countrydaysum['err'] = countrydaysum['err'].fillna(value=0)

sumerr = countrydaysum[countrydaysum['err'] != 0].groupby('Country/Region').agg({'err': lambda x: sum(abs(x))})
sumerr.to_csv("data/sumerr.csv")
#model.to_csv("data/model.csv")

countrydaysum.to_csv("data/countrydaysum.csv")
melted.to_csv("data/melted2.csv")