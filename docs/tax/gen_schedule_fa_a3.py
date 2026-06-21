import pandas as pd, yfinance as yf, sys, subprocess, re
from datetime import timedelta

year = int(sys.argv[1])
cy_s, cy_e = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
fy_s, fy_e = pd.Timestamp(f"{year}-04-01"), pd.Timestamp(f"{year+1}-03-31")

intc = yf.Ticker("INTC").history(start="2022-01-01", end=(fy_e+timedelta(5)).strftime("%Y-%m-%d"))
intc.index = intc.index.tz_localize(None)

rc = {}

def _fetch_sbi_ttbr(date_str):
    """Fetch SBI TT Buy rate from officialforexrates.com for a given YYYY-MM-DD date."""
    token_cmd = "curl -sL 'https://officialforexrates.com/' -c /tmp/fx_ck.txt"
    html = subprocess.run(token_cmd, shell=True, capture_output=True, text=True).stdout
    m = re.search(r'authenticity_token" value="([^"]+)"', html)
    if not m: return None
    token = m.group(1)
    post_cmd = f"curl -sL 'https://officialforexrates.com/' -X POST -b /tmp/fx_ck.txt -d 'authenticity_token={token}&date={date_str}' -H 'Content-Type: application/x-www-form-urlencoded'"
    resp = subprocess.run(post_cmd, shell=True, capture_output=True, text=True).stdout
    rates = re.findall(r'<td>([0-9]+\.[0-9]+)</td>', resp)
    return float(rates[0]) if rates else None

def rate_on_date(dt):
    """SBI TT Buy rate on the given date (falls back to previous days if holiday)."""
    dt = pd.Timestamp(dt)
    key = dt.strftime('%Y-%m-%d')
    if key not in rc:
        for d in range(5):
            day = (dt - timedelta(days=d)).strftime('%Y-%m-%d')
            rate = _fetch_sbi_ttbr(day)
            if rate:
                rc[key] = rate
                break
        else: rc[key] = 85.0
    return rc[key]

def price(dt):
    prior = intc.loc[:pd.Timestamp(dt)]
    return prior.iloc[-1]['Close'] if len(prior) else intc.iloc[0]['Close']

monthly_rates = {}
def rate_for_month(yr, month):
    """Get SBI TTBR for mid-month (used for peak calculation efficiency)."""
    key = f"{yr}-{month:02d}"
    if key not in monthly_rates:
        monthly_rates[key] = rate_on_date(pd.Timestamp(f"{yr}-{month:02d}-15"))
    return monthly_rates[key]

def peak_inr(start, end):
    """Peak value in INR = max(daily_close × monthly_rate) over the range."""
    sub = intc.loc[(intc.index >= start) & (intc.index <= end)]
    if not len(sub): return price(end) * rate_on_date(end)
    best = 0
    for dt, row in sub.iterrows():
        val = row['Close'] * rate_for_month(dt.year, dt.month)
        if val > best: best = val
    return best

# Parse inputs
bs = pd.read_excel(sys.argv[2])
held = []
for _,r in bs.iterrows():
    if pd.isna(r.get("Date Acquired")): continue
    dt = pd.to_datetime(r["Date Acquired"], dayfirst=True)
    if dt > cy_e: continue
    pt = str(r.get("Plan Type","")).strip()
    st = "RSU" if pt=="Rest. Stock" else ("SPP" if pt=="ESPP" else None)
    if not st: continue
    q = r.get("Sellable Qty.",0)
    if pd.isna(q) or int(q)==0: continue
    cb = float(r.get("Est. Cost Basis (per share):",0)) if pd.notna(r.get("Est. Cost Basis (per share):",0)) else 0.0
    held.append({'dt':dt,'type':st,'qty':int(q),'cb':cb})

gl = pd.read_excel(sys.argv[3])
gl = gl[gl['Record Type']=='Sell']
sold = []
for _,r in gl.iterrows():
    try: acq,sdt = pd.to_datetime(r['Date Acquired']),pd.to_datetime(r['Date Sold'])
    except: continue
    pt = str(r.get('Plan Type','')).strip()
    st = 'RSU' if pt=='RS' else ('SPP' if pt=='ESPP' else None)
    if not st: continue
    sold.append({'dt':acq,'type':st,'qty':int(r['Quantity']),
        'cb':float(r['Adjusted Cost Basis Per Share']) if pd.notna(r['Adjusted Cost Basis Per Share']) else 0.0,
        'sale_dt':sdt,'proceeds':float(r['Total Proceeds']) if pd.notna(r['Total Proceeds']) else 0.0})

# Schedule FA A3 — merge held + sold by (date, type)
cy_sold = [s for s in sold if cy_s<=s['sale_dt']<=cy_e]
merged = {}
for h in held:
    key = (h['dt'].strftime('%Y-%m-%d'), h['type'])
    if key not in merged:
        merged[key] = {'dt':h['dt'],'type':h['type'],'held_qty':0,'cb':0,'sold_qty':0,'proceeds':0,'sale_dt':None}
    merged[key]['held_qty'] += h['qty']
    merged[key]['cb'] = h['cb']
for s in cy_sold:
    key = (s['dt'].strftime('%Y-%m-%d'), s['type'])
    if key not in merged:
        merged[key] = {'dt':s['dt'],'type':s['type'],'held_qty':0,'cb':0,'sold_qty':0,'proceeds':0,'sale_dt':None}
    merged[key]['sold_qty'] += s['qty']
    merged[key]['proceeds'] += s['proceeds']
    merged[key]['sale_dt'] = s['sale_dt']
    if merged[key]['cb'] == 0: merged[key]['cb'] = s['cb']

# Closing price and rate for CY end
cp = price(cy_e)
closing_rate = rate_on_date(cy_e)

# E*Trade account total on Dec 31 (includes cash balance)
etrade_total = float(sys.argv[4]) if len(sys.argv) > 4 else None
total_held_shares = sum(m['held_qty'] for m in merged.values())
stock_value_usd = total_held_shares * cp
cash_balance = (etrade_total - stock_value_usd) if etrade_total else 0
closing_inr_total = round((etrade_total if etrade_total else stock_value_usd) * closing_rate)

fa = []
for key in sorted(merged.keys()):
    m = merged[key]
    dt, total_qty = m['dt'], m['held_qty'] + m['sold_qty']
    init = round(total_qty * m['cb'] * rate_on_date(dt))
    pk_stock = total_qty * peak_inr(max(dt, cy_s), cy_e)
    pk_cash = (cash_balance * closing_rate * m['held_qty'] / total_held_shares) if total_held_shares > 0 and m['held_qty'] > 0 else 0
    pk = round(pk_stock + pk_cash)
    cl = round(closing_inr_total * m['held_qty'] / total_held_shares) if total_held_shares > 0 and m['held_qty'] > 0 else 0
    pr = round(m['proceeds'] * rate_on_date(m['sale_dt'])) if m['sold_qty'] > 0 else 0
    fa.append({'Country/Region name':'United States of America','Country Name and Code':'2',
        'Name of entity':'Intel Corporation','Address of entity':'2200 Mission College Blvd Rnb-4-151 SANTA CLARA CA 95054 US','ZIP Code':'95054',
        'Nature of entity':'Public Company','Date of acquiring the interest':dt.strftime('%Y-%m-%d'),
        'Initial value of the investment':init,
        'Peak value of investment during the Period':pk,
        'Closing balance':cl,
        'Total gross amount paid/credited with respect to the holding during the period':0,
        'Total gross proceeds from sale or redemption of investment during the period':pr})
pd.DataFrame(fa).to_csv("Schedule_FA_A3.csv",index=False)

# SPP_RSU_DataEntry (CY)
cy_sold_all = [s for s in sold if cy_s<=s['sale_dt']<=cy_e]
de = [{'Date of acquiring the interest':h['dt'].strftime('%Y-%m-%d'),'Stock Type':h['type'],
       'Number of Shares Purchased':h['qty'],'Sale Date':'','Sale Value (Received in Bank) in INR':''} for h in held]
for s in cy_sold_all:
    sr = rate_on_date(s['sale_dt'])
    de.append({'Date of acquiring the interest':s['dt'].strftime('%Y-%m-%d'),'Stock Type':s['type'],
        'Number of Shares Purchased':s['qty'],'Sale Date':s['sale_dt'].strftime('%Y-%m-%d'),
        'Sale Value (Received in Bank) in INR':round(s['proceeds']*sr,2)})
de.sort(key=lambda x:(x['Date of acquiring the interest'],x['Sale Date']!=''))
pd.DataFrame(de).to_csv("SPP_RSU_DataEntry.csv",index=False)

# Tax Computation (FY)
fy_sold = [s for s in sold if fy_s<=s['sale_dt']<=fy_e]
tx = []
for s in fy_sold:
    br,sr = rate_on_date(s['dt']),rate_on_date(s['sale_dt'])
    ci,pi = round(s['qty']*s['cb']*br,2),round(s['proceeds']*sr,2)
    days = (s['sale_dt']-s['dt']).days
    tx.append({'Record Type':'Sell','Symbol':'INTC','Plan Type':s['type'],'Quantity':s['qty'],
        'Date Acquired':s['dt'].strftime('%Y-%m-%d'),'Date Sold':s['sale_dt'].strftime('%Y-%m-%d'),
        'Calculated Cost Basis':ci,'Total Proceeds':pi,'USD_INR_Buy':round(br,4),'USD_INR_Sell':round(sr,4),
        'Buy Price in INR':round(s['cb']*br,2),'Sell Price in INR':round((s['proceeds']/s['qty'])*sr,2),
        'Hold Duration':days,'Hold Type':'LTCG' if days>365 else 'STCG','Gain/Loss':round(pi-ci,2)})
pd.DataFrame(tx).to_csv("Tax_Computation.csv",index=False)

print(f"CY {year} | INTC Dec31: ${cp:.2f} | Rate(Dec31): {closing_rate:.2f}")
print(f"Schedule_FA_A3.csv: {len(fa)} | SPP_RSU_DataEntry.csv: {len(de)} | Tax_Computation.csv: {len(tx)}")
print(f"Capital Gains: ₹{sum(r['Gain/Loss'] for r in tx):,.2f}")
