# KeckAoTelemetry
Tools and tips to process Keck Observatory AO Telemetry

## NGWFC
- The Keck Observatory NGWFC writes a telemetry stream into a postgres database.  
- The live database can store about 10 nights worth of telemetry
- A library of database commands was written to interface to IDL
- Extracted telemetry corresponding to NIRC2 images is dumped to IDL ".sav" files for archiving
- Basic description: https://www2.keck.hawaii.edu/optics/ao/ngwfc/trsqueries.html
