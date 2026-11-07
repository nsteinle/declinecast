-- One typed row per record in the yearly ECMC production files.
-- Columns are read by name, not position: the 2025 file renames three columns and moves
-- FlaredVented ahead of WaterProduced, so a positional read would take flared gas for water.
-- The 2025 file also writes missing values as the text NULL and dates in ISO format, where the
-- earlier files leave the field empty and write MM/DD/YYYY.
SELECT
    '05' || lpad(trim(ApiCountyCode), 3, '0') || lpad(trim(ApiSequenceNumber), 5, '0')
         || lpad(coalesce(trim(ApiSidetrack), '00'), 2, '0')        AS api,
    CAST(ReportYear AS INTEGER)                                      AS year,
    CAST(ReportMonth AS INTEGER)                                     AS month,
    TRY_CAST(DaysProduced AS INTEGER)                                AS days,
    TRY_CAST(OilProduced AS DOUBLE)                                  AS oil_bbl,
    TRY_CAST(GasProduced AS DOUBLE)                                  AS gas_mcf,
    TRY_CAST(WaterProduced AS DOUBLE)                                AS water_bbl,
    upper(trim(FormationCode))                                       AS formation,
    upper(trim(WellStatus))                                          AS status,
    trim(WellStatus) <> upper(trim(WellStatus))                      AS status_not_upper,
    trim(OpName)                                                     AS operator,
    trim(OpNumber)                                                   AS op_num,
    coalesce(try_strptime(AcceptedDate, '%m/%d/%Y %H:%M:%S'),
             try_strptime(AcceptedDate, '%Y-%m-%d %H:%M:%S.%g'))     AS accepted,
    upper(coalesce(trim(Revised), '')) IN ('Y', 'TRUE', '1')         AS revised,
    DocNum                                                           AS doc_num,
    CAST(regexp_extract(parse_filename(filename), '(\d{{4}})', 1) AS INTEGER) AS file_year
FROM read_csv([{files}], all_varchar = true, union_by_name = true, header = true,
              nullstr = ['', 'NULL'], filename = true)
