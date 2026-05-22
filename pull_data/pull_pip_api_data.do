pip, year(2023 2025) fillgaps nowcast povline(2.15 3) ppp(2017) clear
gen ppp = 2017
tempfile r17
save `r17'
pip, year(2023 2025) fillgaps nowcast povline(2.15 3) ppp(2021) clear
gen ppp = 2021
append using `r17'

local today = string(year(date(c(current_date), "DMY"))) + ///
              string(month(date(c(current_date), "DMY")), "%02.0f") + ///
              string(day(date(c(current_date), "DMY")), "%02.0f")
local savefile "/data/eop/compiled_country_data/from_pip_api/poverty_data_accessed_`today'.dta"
save "`savefile'", replace