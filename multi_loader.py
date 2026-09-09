"""
multi_loader.py  ·  Unified Indian bank/UPI statement loader
─────────────────────────────────────────────────────────────
Flow (same for CSV and Excel):
  File received
    ↓ Read ALL rows — no header assumption
    ↓ find_transaction_header() — scans rows, finds real table
    ↓ Skip bank/customer info above the table
    ↓ Normalize column names
    ↓ detect_source() — bank / PhonePe / GPay / Paytm
    ↓ Parse transactions → clean DataFrame

Supports:
  Bank statements  : SBI, HDFC, ICICI, Axis, Kotak, PNB, Canara, BOI …
  UPI apps         : PhonePe, Google Pay, Paytm
  Generic CSV/Excel: any file with Date + Amount + Description columns
"""

import io
import csv
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


# ════════════════════════════════════════════════════════════════
# CATEGORY MAP
# ════════════════════════════════════════════════════════════════
CATEGORY_KEYWORDS = {
    'Food & Dining'  : ['swiggy','zomato','mcdonalds','dominos','cafe',
                        'restaurant','food','pizza','burger','blinkit',
                        'dunzo','instamart','eat','dining','bakery'],
    'Transport'      : ['uber','ola','rapido','irctc','petrol','fuel',
                        'auto','bus','metro','cab','taxi','parking','fastag'],
    'Shopping'       : ['amazon','flipkart','myntra','meesho','ajio',
                        'mall','store','shop','market','retail','nykaa',
                        'zepto','bigbasket','dmart','jiomart'],
    'Bills & Rent'   : ['electricity','jio','airtel','broadband','rent',
                        'water','gas','maintenance','society','bill',
                        'recharge','insurance','lic','bsnl'],
    'Entertainment'  : ['netflix','prime','spotify','hotstar','bookmyshow',
                        'pvr','cinema','movie','game','youtube','disney'],
    'Health'         : ['pharmacy','apollo','medplus','doctor','hospital',
                        'clinic','gym','fitness','yoga','cult','1mg','netmeds'],
    'Education'      : ['udemy','coursera','nptel','unacademy','book',
                        'stationery','school','college','tuition','byju'],
    'Travel'         : ['indigo','airindia','spicejet','oyo','airbnb',
                        'hotel','resort','flight','train','makemytrip',
                        'goibibo','cleartrip'],
    'UPI Transfer'   : ['upi','neft','rtgs','imps','transfer','sent to',
                        'paid to','@ok','@ybl','@upi','@paytm','@ibl'],
}

def categorise(description):
    desc = str(description).lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(k in desc for k in kws):
            return cat
    return 'Others'


# ════════════════════════════════════════════════════════════════
# STEP 1 — READ ALL ROWS (no header assumption)
# ════════════════════════════════════════════════════════════════
def read_raw(file_input, is_excel=False):
    """
    Read every row without assuming the first row is the header.

    For CSV, use Python's csv reader instead of pandas on_bad_lines='skip'.
    Bank/UPI exports often contain metadata rows with fewer columns than the
    transaction table; those rows must not cause the real transaction rows
    to be discarded.
    """
    if is_excel:
        if isinstance(file_input, bytes):
            src = io.BytesIO(file_input)
        elif isinstance(file_input, io.BytesIO):
            file_input.seek(0)   # always rewind before reading
            src = file_input
        else:
            src = file_input
        return pd.read_excel(src, header=None, dtype=str)

    # Obtain raw bytes/text.
    if isinstance(file_input, str):
        raw_bytes = Path(file_input).read_bytes()
    elif isinstance(file_input, bytes):
        raw_bytes = file_input
    elif isinstance(file_input, io.BytesIO):
        file_input.seek(0)
        raw_bytes = file_input.read()
    else:
        raw_bytes = str(file_input).encode('utf-8')

    # Decode robustly.
    text = None
    for enc in ('utf-8-sig', 'utf-8', 'cp1252', 'latin1'):
        try:
            text = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        raise ValueError("Could not decode the CSV file.")

    # Parse records without dropping rows of different widths.
    try:
        rows = list(csv.reader(io.StringIO(text)))
    except Exception as e:
        raise ValueError(f"Could not parse the CSV file: {e}")

    if not rows:
        return pd.DataFrame()

    width = max(len(r) for r in rows)
    padded = [r + [''] * (width - len(r)) for r in rows]
    return pd.DataFrame(padded, dtype=str)


# ════════════════════════════════════════════════════════════════
# STEP 2 — FIND TRANSACTION HEADER ROW
# ════════════════════════════════════════════════════════════════
# Known transaction column names across Indian banks
HEADER_ALIASES = {
    'date': {
        'date','txn date','transaction date','value date',
        'posting date','transaction dt','txn dt','trans date',
        'transaction posted date',
    },
    'description': {
        'description','particulars','narration','details',
        'transaction details','transaction description',
        'transaction remarks','remarks','reference',
        'paid to','paid to/from','merchant','recipient',
        'beneficiary','transaction note','notes',
    },
    'debit': {
        'debit','withdrawal','withdrawal amt','withdrawal amount',
        'debit amount','dr','dr amount','debit amt',
        'amount(dr)','dr.','withdrawals',
    },
    'credit': {
        'credit','deposit','deposit amount','credit amount',
        'cr','cr amount','credit amt','amount(cr)','cr.',
    },
    'amount': {
        'amount','transaction amount','txn amount','amount (inr)',
        'total amount','net amount',
    },
    'id': {
        'transaction id','txn id','utr','reference no','ref no',
        'cheque no','chq no','sl no','sr no',
    },
    'balance': {
        'balance','closing balance','available balance',
        'running balance','bal',
    },
}

# Rows that look like account metadata — skip them
METADATA_INDICATORS = [
    'branch name','account no','account number','customer name',
    'ifsc','micr','clear balance','opening balance','closing balance',
    'statement period','account type','nominee','address','phone',
    'email','pan','aadhar','mobile','city','state','product code',
    'cif no','cif number','sol id','currency','scheme code',
    'account holder','generated on','period',
    'transaction statement','duration','statement generated',
]

def _norm(v):
    """Lowercase + collapse whitespace."""
    return re.sub(r'\s+', ' ', str(v).strip().lower())

def find_transaction_header(raw_df, max_scan_rows=80):
    """
    Scan rows top-to-bottom, score each against known transaction
    column keywords.  Return a DataFrame whose columns come from the
    best-scoring row and whose rows are the transactions below it.

    Returns None if no suitable header is found.
    """
    sample = raw_df.head(max_scan_rows)
    best_idx, best_score = None, -1

    for idx, row in sample.iterrows():
        values = [_norm(v) for v in row.tolist()
                  if str(v).strip() not in ('', 'nan', 'none', 'nat')]

        # Skip rows that are clearly bank metadata
        row_text = ' '.join(values)
        if any(m in row_text for m in METADATA_INDICATORS):
            continue

        # Score the row
        score = 0
        for v in values:
            if v in HEADER_ALIASES['date']:          score += 4
            elif v in HEADER_ALIASES['description']: score += 4
            elif v in HEADER_ALIASES['debit']:       score += 3
            elif v in HEADER_ALIASES['credit']:      score += 3
            elif v in HEADER_ALIASES['amount']:      score += 3
            elif v in HEADER_ALIASES['id']:          score += 2
            elif v in HEADER_ALIASES['balance']:     score += 1

        # Bonus: row has BOTH a date column AND a money column
        has_date  = any(v in HEADER_ALIASES['date'] for v in values)
        has_money = any(
            v in HEADER_ALIASES['debit']
            or v in HEADER_ALIASES['credit']
            or v in HEADER_ALIASES['amount']
            for v in values
        )
        if has_date and has_money:
            score += 5

        if score > best_score:
            best_score, best_idx = score, idx

    if best_idx is None or best_score < 7:
        return None  # no transaction table found

    # Build DataFrame: header = that row, data = everything below
    header = [_norm(v) for v in raw_df.iloc[best_idx].tolist()]
    data   = raw_df.iloc[best_idx + 1:].copy()
    data.columns = header

    # Drop unnamed / empty columns
    data = data.loc[:, [c for c in data.columns
                        if c and c not in ('nan','none','')]]
    data = data.dropna(how='all').reset_index(drop=True)
    return data


# ════════════════════════════════════════════════════════════════
# STEP 3 — DETECT SOURCE
# ════════════════════════════════════════════════════════════════
def detect_source(df):
    cols = ' '.join(_norm(c) for c in df.columns)

    # PhonePe: has Transaction Id + Amount (INR), or Paid To, or UTR
    if (
        ('transaction id' in cols and 'amount (inr)' in cols)
        or ('transaction id' in cols and 'utr' in cols)
        or ('transaction type' in cols and 'credit/debit instrument' in cols)
        or ('transaction details' in cols and 'transaction id' in cols)
    ):
        return 'phonepe'

    if 'paid to' in cols or 'paid to/from' in cols or 'wallet transaction' in cols:
        return 'phonepe'

    if 'product' in cols and ('google' in cols or 'gpay' in cols):
        return 'gpay'

    if 'comment' in cols and ('credit/debit' in cols or 'credited/debited' in cols):
        return 'paytm'

    if any(c in df.columns for c in [
        'debit','withdrawal','dr','withdrawal amt',
        'narration','particulars','chq/ref number',
        'debit amount','dr amount','credit','deposit',
    ]):
        return 'bank'

    return 'generic'


# ════════════════════════════════════════════════════════════════
# STEP 4 — PARSE TRANSACTIONS
# ════════════════════════════════════════════════════════════════
def _find_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def _parse_rows(df, date_cands, desc_cands, debit_cands,
                credit_cands=None, type_cands=None,
                status_cands=None, source='Bank'):
    cols       = df.columns.tolist()
    date_col   = _find_col(cols, date_cands)
    desc_col   = _find_col(cols, desc_cands)
    debit_col  = _find_col(cols, debit_cands)
    credit_col = _find_col(cols, credit_cands or [])
    type_col   = _find_col(cols, type_cands  or [])
    stat_col   = _find_col(cols, status_cands or [])

    if not date_col or not debit_col:
        return []

    rows = []
    for _, r in df.iterrows():
        # Skip failed UPI transactions
        if stat_col:
            s = str(r.get(stat_col, '')).lower()
            if s and s not in ('success','completed','successful',''):
                continue
        # Skip credit rows
        if type_col:
            t = str(r.get(type_col, '')).lower()
            if 'credit' in t or t == 'cr':
                continue

        amt = _clean_amt(r.get(debit_col, 0))
        if amt <= 0 and credit_col:
            amt = _clean_amt(r.get(credit_col, 0))
        if amt <= 0:
            continue

        date = _parse_date(str(r.get(date_col, '')))
        if not date:
            continue

        rows.append({
            'Date'       : date,
            'Description': str(r.get(desc_col, source)).strip()
                           if desc_col else source,
            'Amount'     : amt,
            'Type'       : 'Debit',
            'Source'     : source,
        })
    return rows

def parse_bank(df):
    return _parse_rows(df,
        date_cands  = ['date','txn date','transaction date','value date',
                       'posting date','transaction dt','trans date',
                       'transaction posted date'],
        desc_cands  = ['description','particulars','narration','details',
                       'transaction details','transaction description',
                       'transaction remarks','remarks','reference'],
        debit_cands = ['debit','withdrawal','withdrawal amt','debit amount',
                       'dr','dr amount','debit amt','withdrawals','amount'],
        credit_cands= ['credit','deposit','credit amount','cr','cr amount'],
        source      = 'Bank')

def parse_phonepe(df):
    return _parse_rows(df,
        date_cands   = ['date','transaction date','time'],
        desc_cands   = [
            'transaction details','paid to','paid to/from',
            'to','merchant','description','details','narration',
            'remarks','beneficiary',
        ],
        debit_cands  = ['amount (inr)','amount','transaction amount','debit'],
        type_cands   = ['transaction type','type','dr/cr'],
        status_cands = ['status','transaction status'],
        source       = 'PhonePe')

def parse_gpay(df):
    return _parse_rows(df,
        date_cands   = ['date','time','transaction date'],
        desc_cands   = ['description','merchant','to','details','note'],
        debit_cands  = ['amount','transaction amount','debit'],
        status_cands = ['status','transaction status'],
        source       = 'GPay')

def parse_paytm(df):
    return _parse_rows(df,
        date_cands  = ['date','transaction date'],
        desc_cands  = ['comment','description','merchant','details'],
        debit_cands = ['amount','transaction amount','debit'],
        type_cands  = ['credit/debit','type','dr/cr','credited/debited'],
        source      = 'Paytm')

def parse_generic(df):
    cols     = df.columns.tolist()
    date_col = next((c for c in cols if 'date' in c or 'time' in c), None)
    amt_col  = next((c for c in cols if 'amount' in c or 'debit' in c
                     or 'price' in c or 'value' in c), None)
    desc_col = next((c for c in cols if any(k in c for k in
                     ['desc','name','particular','narration',
                      'detail','merchant','remark'])), None)
    if not date_col or not amt_col:
        raise ValueError(
            f"Could not auto-detect columns.\n"
            f"Columns found: {cols}\n"
            f"Your file must have Date, Amount/Debit, Description columns."
        )
    return _parse_rows(df,
        date_cands  = [date_col],
        desc_cands  = [desc_col] if desc_col else [],
        debit_cands = [amt_col],
        source      = 'Other')


# ════════════════════════════════════════════════════════════════
# STEP 5 — FINALISE DataFrame
# ════════════════════════════════════════════════════════════════
def rows_to_df(rows):
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df[df['Amount'] > 0].dropna(subset=['Date','Amount'])
    df['Category'] = df['Description'].apply(categorise)
    df['Month']    = df['Date'].dt.strftime('%B %Y')
    df['Month_Num']= df['Date'].dt.month
    df['Year']     = df['Date'].dt.year
    return df.sort_values('Date').reset_index(drop=True)


# ════════════════════════════════════════════════════════════════
# PUBLIC API — single entry point for both CSV and Excel
# ════════════════════════════════════════════════════════════════
def load_file(file_input, password=None, filename=''):
    """
    Universal loader — same pipeline for CSV and Excel.

    Args:
        file_input : file path (str), BytesIO, or raw bytes
        password   : optional password for protected Excel files
        filename   : original filename (used to detect Excel vs CSV)

    Returns:
        Clean transaction DataFrame
    """
    is_excel = str(filename).lower().endswith(('.xlsx', '.xls'))

    # ── Unlock Excel if needed ─────────────────────────────────────────────
    if is_excel and password:
        file_input, err = unlock_excel(
            file_input if isinstance(file_input, bytes)
            else file_input.read(), password)
        if err:
            raise ValueError(err)

    # ── Step 1: Read ALL rows ──────────────────────────────────────────────
    raw = read_raw(file_input, is_excel=is_excel)

    # ── Step 2: Find transaction header ───────────────────────────────────
    df_prepared = find_transaction_header(raw)
    if df_prepared is None:
        raise ValueError(
            "Could not find a transaction table in this file.\n"
            "Make sure this is a bank statement downloaded from netbanking.\n"
            "Try downloading as CSV/Excel from your bank portal."
        )

    # ── Step 3: Detect source ──────────────────────────────────────────────
    source = detect_source(df_prepared)

    # ── Step 4: Parse ─────────────────────────────────────────────────────
    parsers = {
        'bank'    : parse_bank,
        'phonepe' : parse_phonepe,
        'gpay'    : parse_gpay,
        'paytm'   : parse_paytm,
        'generic' : parse_generic,
    }
    rows = parsers[source](df_prepared)

    # ── Step 5: Build DataFrame ────────────────────────────────────────────
    df = rows_to_df(rows)
    if df is None or len(df) == 0:
        raise ValueError(
            "No debit transactions found after parsing.\n"
            "Check that your file contains expense/withdrawal rows."
        )
    print(f"  ✅ {len(df):,} transactions | source={source}")
    return df


# Backward-compatible alias used by notebook cells
def load_single_csv(filepath):
    return load_file(filepath, filename=str(filepath))

def load_multiple_csvs(file_list):
    """Load and merge multiple files, deduplicating shared transactions."""
    all_dfs = []
    for f in file_list:
        fname = f.name if hasattr(f, 'name') else str(f)
        print(f"\nLoading: {fname}")
        try:
            src = f if isinstance(f, str) else f.read()
            df  = load_file(src, filename=fname)
            all_dfs.append(df)
        except Exception as e:
            print(f"  ❌ Skipped: {e}")

    if not all_dfs:
        raise ValueError("No valid files could be loaded.")

    combined = pd.concat(all_dfs, ignore_index=True).sort_values('Date')
    before   = len(combined)
    combined = _deduplicate(combined)
    if before != len(combined):
        print(f"\n🔁 Removed {before-len(combined)} duplicates")
    return combined.reset_index(drop=True)

def _deduplicate(df):
    df = df.copy()
    df['_key'] = df['Date'].dt.strftime('%Y-%m-%d') + '_' + df['Amount'].astype(str)
    df['_len'] = df['Description'].str.len()
    df = (df.sort_values('_len', ascending=False)
            .drop_duplicates('_key', keep='first')
            .drop(columns=['_key','_len'])
            .sort_values('Date')
            .reset_index(drop=True))
    return df


# ════════════════════════════════════════════════════════════════
# Excel password unlock
# ════════════════════════════════════════════════════════════════
def unlock_excel(file_bytes, password):
    """Decrypt a password-protected Excel workbook in memory."""
    try:
        import msoffcrypto

        enc = io.BytesIO(
            file_bytes if isinstance(file_bytes, bytes)
            else file_bytes.read()
        )
        dec = io.BytesIO()

        # IMPORTANT: load the key and decrypt using the SAME OfficeFile object.
        office_file = msoffcrypto.OfficeFile(enc)
        office_file.load_key(password=str(password))
        office_file.decrypt(dec)

        dec.seek(0)
        return dec, None

    except Exception as e:
        return None, f"Wrong password or unsupported format: {e}"


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════
def _clean_amt(text):
    if not text or str(text).strip() in ('', 'nan', 'NaN', '-', 'Nil', 'nil'):
        return 0.0
    text = re.sub(r'[₹$,\s]', '', str(text)).replace('(', '-').replace(')', '')
    try:
        return abs(float(text))
    except ValueError:
        return 0.0

def _parse_date(text):
    fmts = [
        '%d/%m/%Y','%d-%m-%Y','%Y-%m-%d','%d/%m/%y','%d-%m-%y',
        '%d %b %Y','%d %B %Y','%Y/%m/%d','%m/%d/%Y',
        '%Y-%m-%d %H:%M:%S','%d/%m/%Y %H:%M:%S',
        '%d-%m-%Y %H:%M:%S','%d %b %Y %H:%M:%S',
        '%b %d, %Y','%B %d, %Y','%d/%m/%Y %H:%M',
    ]
    text = str(text).strip()
    for f in fmts:
        try:
            return datetime.strptime(text, f)
        except ValueError:
            continue
    try:
        return pd.to_datetime(text, dayfirst=True)
    except Exception:
        return None
