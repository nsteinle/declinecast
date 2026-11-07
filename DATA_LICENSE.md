# Bundled data

`src/declinecast/data/wells.parquet` and `src/declinecast/data/months.parquet`

- **Source:** Colorado Energy & Carbon Management Commission (ECMC, formerly the Colorado Oil
  and Gas Conservation Commission), downloadable data:
  https://ecmc.colorado.gov/data-maps-reports/downloadable-data-documents
- **Files used:** the yearly production reports `2013_prod_reports.zip` to
  `2025_prod_reports.zip` and the well master `WELLS_SHP.ZIP`, downloaded from
  `https://ecmc.state.co.us/documents/data/downloads/`. The SHA-256 of each is recorded in
  `src/declinecast/data/panel.json`.
- **Data dictionary:**
  https://ecmc.state.co.us/documents/data/downloads/production/production_record_data_dictionary.htm
- **Credit:** production and well data courtesy of the Colorado Energy & Carbon Management
  Commission. ECMC does not endorse this package, and the extract is not an official ECMC
  product. Errors in the derivation are mine.

## Terms

ECMC publishes these files for public download without a reuse licence that I could find
(checked 17 October 2026: the site's General Notices page is a privacy policy, its Legal
Notices page is a DMCA procedure, and the footer reads "© 2026 State of Colorado"). Production
volumes, dates and well locations are facts reported to a regulator. Under US law facts are not
protected by copyright, and a compilation is protected only in its original selection or
arrangement (*Feist Publications v. Rural Telephone Service*, 499 U.S. 340 (1991)). I am not a
lawyer, and this is my reading, not legal advice.

On that basis the package:

- does **not** redistribute the raw files. It ships only a derived extract, built by its own
  code with its own selection and layout;
- credits ECMC here, in `CITATION.cff` and in the README;
- downloads the raw files only when you ask (`declinecast fetch`), from the published URLs, one
  request at a time, with no crawling.

If ECMC asks for the extract to be removed, the package will switch to fetch-only: the
extract leaves the package and `declinecast panel` rebuilds it locally from files you download
yourself.

## How the extract was made

`declinecast panel` (DuckDB SQL in `src/declinecast/sql/`) did the following, and
`docs/FINDINGS.md` section 1 counts every step:

1. read every yearly file by column name, because the 2025 file renames three columns and moves
   one, and writes missing values as the text `NULL`;
2. kept one version of each (well, month, formation) record: the most recently accepted for the
   final panel, the earliest for the as-first-reported columns (`oil_bbl_first`,
   `days_first`), because the 2025 file re-exports about 240,000 earlier records;
3. summed formation rows into one row per well and month (commingled wells report once per
   formation);
4. kept the horizontal Niobrara and Codell wells (primary formation NBRR, NB-CD, CODL, NIOB or
   CDLL) whose first producing month is in 2014 or later, with at least 12 producing months, a
   lateral (measured minus true vertical depth) of at least 3,000 ft, a spud date in 2012 or
   later, and possible geometry (true vertical depth below measured depth): 8,344 wells;
5. kept, per well: API number, first and last producing month, producing months, primary
   formation, first and last operator as filed, spud date, depths, lateral, latitude and
   longitude, township, range, section, pad (location ID) and field; and per well and month:
   oil, gas, water, producing days, status, the revision flag, the number of versions filed,
   the number of formation rows, and the first-filed oil and days.

Nothing else was changed. The extract's SHA-256 is recorded in `panel.json` and checked every
time it is loaded.
