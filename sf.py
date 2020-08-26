#!/usr/bin/python
# -*- coding: utf-8 -*-
import quandl
import numpy as np
import pandas as pd
import matplotlib as mpl
from datetime import datetime
from datetime import timedelta, date
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def filter_empty_datapoints(df):
    indices = df[df.sum(axis=1) == 0].index
    df = df.drop(indices)
    return df


def get_quandl_data(quandl_id):
    quandl.ApiConfig.api_key = 'API_KEY'
    df = quandl.get(quandl_id, returns='pandas')
    df = filter_empty_datapoints(df)
    return df


def days_between(d1, d2):
    d1 = datetime.strptime(d1, '%Y-%m-%d')
    d2 = datetime.strptime(d2, '%Y-%m-%d')
    return abs((d2 - d1).days)


def btcSupplyAtBlock(b):
    if b >= 33 * 210000:
        return 20999999.9769
    else:
        reward = 50e8
        supply = 0
        y = 210000
        while b > y - 1:
            supply = supply + y * reward
            reward = int(reward / 2.0)
            b = b - y
        supply = supply + b * reward
        return ((supply + reward) / 1e8, reward / 1e8)


totbtc = get_quandl_data('BCHAIN/TOTBC')['2010-09-01':]
cap = get_quandl_data('BCHAIN/MKTCP')['2010-09-01':]
btc = get_quandl_data('BCHARTS/KRAKENUSD')['2010-09-01':]

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
figure(num=None, figsize=(16, 16), dpi=200)

totbtc['flow'] = totbtc['Value'].diff(periods=14)
totbtc['flow1y'] = totbtc['Value'].diff(periods=365)
totbtc = filter_empty_datapoints(totbtc)

totbtc['sf'] = totbtc['Value'] / totbtc['flow']
totbtc['sf1y'] = totbtc['Value'] / totbtc['flow1y']
totbtc['cap'] = cap['Value']

(fig, ax1) = plt.subplots()

hdates = [
    '2009-01-01',
    '2012-11-28',
    '2016-07-09',
    '2020-05-11',
    '2024-05-01',
    '2028-05-01',
    '2032-05-01',
    ]

h = pd.DataFrame(columns=['Date', 'ds_bfr_hlvng', 'sf', 'cap'])

for i in range(1, len(hdates)):
    date = hdates[i]
    for (index, row) in totbtc[:date].iterrows():
        if index < datetime.strptime(date, '%Y-%m-%d') and index \
            > datetime.strptime(hdates[i - 1], '%Y-%m-%d'):
            ds_btwn = days_between(str(index.date()), date)
            h = h.append({
                'Date': index.strftime('%Y-%m-%d'),
                'ds_bfr_hlvng': ds_btwn,
                'sf': row['sf'],
                'cap': row['cap'],
                }, ignore_index=True)

h = filter_empty_datapoints(h)

fig = plt.figure(num=None, figsize=(16, 16), dpi=200)

ls = 18
ax = plt.gca()
ax.tick_params(labelsize=ls)

sc = ax.scatter(
    h['sf'],
    h['cap'],
    c=h['ds_bfr_hlvng'],
    alpha=0.9,
    cmap='gist_rainbow',
    s=15,
    zorder=1,
    )
ax.plot(
    h['sf'],
    h['cap'],
    c='#474747',
    alpha=0.6,
    zorder=2,
    linewidth=0.2,
    )
ax.set_title('Stock-to-Flow & Market Value', fontsize=ls)
ax.set_xlabel('log(Stock-to-Flow)', fontsize=ls)
ax.set_ylabel('log(Market Value)', fontsize=ls)
ax.set_yscale('log')
ax.set_xscale('log')

locator = mpl.ticker.MaxNLocator(nbins=6)

cbar = plt.colorbar(sc)
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.tick_params(labelsize=ls)
cbar.ax.set_ylabel('Days before halving', fontsize=ls)

ax.text(
    0.1,
    0.95,
    '$SF = \\frac{S}{F_p}$\np = 14d',
    ha='center',
    va='center',
    style='italic',
    fontsize=ls,
    transform=ax.transAxes,
    bbox=dict(facecolor='none', edgecolor='black', pad=10.0),
    )
fig.tight_layout()

plt.savefig('BTC-SF-MV-2.png', facecolor='#fff3ea', edgecolor='#fff3ea')
plt.savefig('BTC-SF-MV.png')
plt.show()
plt.clf()

d = []
ep = []
genesis = '2009-01-01'
start_date = datetime.strptime('2011-09-01', '%Y-%m-%d')
for date in (start_date + timedelta(n) for n in range(9000)):
    block = days_between(genesis, date.strftime('%Y-%m-%d')) * 24 * 6
    (supply, reward) = btcSupplyAtBlock(block)
    d.append(date)
    sf = supply / (365 * 24 * 6 * reward)
    ep.append(np.exp(12.7598) * sf ** 4.1167 / supply)

fig = plt.figure(num=None, figsize=(16, 16), dpi=200)
plt.title('Stock-to-flow Estimated & Real Price', fontsize=ls)
plt.xlabel('Date', fontsize=ls)
plt.ylabel('ln(Price, USD)', fontsize=ls)
plt.yscale('log')
plt.plot(totbtc.index, totbtc['cap'] / totbtc['Value'],
         color='royalblue', label='BTCUSD')
plt.plot(totbtc.index, np.exp(12.7598) * totbtc['sf1y'] ** 4.1167
         / totbtc['Value'], color='red',
         label='Expected price based on real stock and flow')
plt.plot(d, ep, color='cornflowerblue',
         label='Expected price based on calculated stock and flow')
ax.text(
    0.5,
    0.5,
    'github.com/pyzhyk/sf',
    horizontalalignment='right',
    verticalalignment='center',
    transform=ax1.transAxes,
    fontsize=20,
    bbox=dict(facecolor='#e1e1e1', alpha=0.4, edgecolor='black',
              pad=10.0),
    )
plt.legend(fontsize=ls)
plt.tick_params(labelsize=ls)
plt.grid(True, which='both')
fig.tight_layout()
plt.savefig('BTC-SF-Exp_Price-Price-2.png', facecolor='#fff3ea',
            edgecolor='#fff3ea')
plt.savefig('BTC-SF-Exp_Price-Price.png')
plt.show()

