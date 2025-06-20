# Data Dictionary

## Year_Month
Time period for the data point in YYYY-MM format

## Territory_Upper
Geographic territory code representing sales regions (e.g., A03, B01)

## Segment
Business segment categorization: Commercial (business/industrial customers), Residential (home/consumer customers), or Other

## Net Ordered Quantity
The final quantity ordered after adjusting for cancellations and modifications

## Gross Sales Amount
Total sales amount before adjustments, discounts, or returns in currency

## Net Invoiced Amount
Final invoiced amount after all adjustments, discounts, and returns in currency

## # of Orders
Total count of distinct orders placed in the given period

## Number of Bookings
Count of orders that were successfully booked/confirmed

## Win_Rate
Percent of quotes that turns into sales, calculated as (Number of orders with 100% booking rate) / (Total number of orders). Ranges from 0 to 1, with 1 indicating 100% success

## Unit Selling Price
List price per unit of product before any discounts or adjustments

## Discount Amount
Total monetary value of discounts applied to orders in currency

## % Order Discount
Percentage discount applied to orders, calculated as (Discount Amount / Gross Sales Amount) * 100

## Avg_Products_per_Account
Average number of different products per customer account

## Max_Products_per_Account
Maximum number of different products any account ordered

## Industry_Qty
The total quantity of water heaters sold by the entire industry (RWH + CWH) for the given month, territory, and segment

## Product_Count
Number of unique products sold in each territory-segment-month combination (5 means territory sells 5 different products)

## Product_Concentration
How dependent sales are on top products (0-1) (higher values (closer to 1) indicate dependence on fewer products e.g.,  0.8 means 80% of sales come from top product)

## Top_Product
Name of the highest-selling product by quantity

## Top_Product_Code
Numerical encoding of the top product (for predictive modeling)

## Primary_State
State where territory has highest sales volume

## Primary_City
City where territory has highest sales volume

## Primary_County
County where territory has highest sales volume

## Location_Count
Number of different cities in territory

## Primary_State_Full
Description for  Primary_State_Full needs to be added

## Top_Customer
Name of customer with highest order volume in territory

## Top_Customer_Code
Numerical ID for top customer (for modeling)

## Market_Share
Percentage of total industry sales achieved by the company, calculated as (Net Ordered Quantity / Industry_Qty) * 100

## CPI
Changes in prices overtime

## Count
Total count of inventory for the given month

## Make or Buy
indicates whether the products is manufactured in-house (Make) or sourced externally (buy)

## Criteria 4
Categorization of inventory based on specific business criteria

## Plant Location Number
Unique indentifier for the manufacturing or storage facility

## Sales Org Name
Name of the sales organization responsible for the inventory

## Item Status
Description for  Item Status needs to be added

## United_States_Total_completed
Total number of new privately-owned housing units completed in the entire United States, measured in thousands of unit (a value of 1,000 = 1,000,000)

## United_States_1Unit_completed
Number of single-family homes (standalone houses) completed in the United States, measured in thousands of units

## United_States_2to4Units_completed
Number of housing units completed that are part of small multi-family buildings, such as duplexes, triplexes, or fourplexs, measured in thousands of units

## United_States_5units_or_more_completed
Number of housing unit completed in larger apartment buildings or complexes, measured in thousands of units

## Northeast_Total_completed
Total Housing units completed in the Northeast region

## Northeast_1Unit_completed
Single-unit housing units completed in the Northeast region

## Midwest_Total_completed
Total Housing units completed in the Midwest region

## Midwest_1Unit_completed
Single-unit housing units completed in the Midwest region

## South_Total_completed
Total Housing units completed in the South region

## South_1Unit_completed
Single-unit housing units completed in the South region

## West_Total_completed
Total Housing units completed in the West region

## West_1Unit_completed
Single-unit housing units completed in the West region

## United_States_Total_under_construction
Total number of new privately-owned housing units under construction in the entire United States, measured in thousands of unit

## United_States_1Unit_under_construction
Number of single-family homes (standalone houses) under construction in the United States, measured in thousands of units

## United_States_2to4Units_under_construction
Number of housing units under construction that are part of small multi-family buildings, such as duplexes, triplexes, or fourplexs, measured in thousands of units

## United_States_5units_or_more_under_construction
Number of housing unit under construction in larger apartment buildings or complexes, measured in thousands of units

## Northeast_Total_under_construction
Total number of new privately-owned housing units under construction in the Northeast region, measured in thousands of unit

## Northeast_1Unit_under_construction
Number of single-family homes (standalone houses) under construction in the Northeast region, measured in thousands of units

## Midwest_Total_under_construction
Total number of new privately-owned housing units under construction in the entire Midwest region, measured in thousands of unit

## Midwest_1Unit_under_construction
Number of single-family homes (standalone houses) under construction in the Midwest region, measured in thousands of units

## South_Total_under_construction
Total number of new privately-owned housing units under construction in the South region, measured in thousands of unit

## South_1Unit_under_construction
Number of single-family homes (standalone houses) under construction in the South regoin, measured in thousands of units

## West_Total_under_construction
Total number of new privately-owned housing units under construction in the West region, measured in thousands of unit

## West_1Unit_under_construction
Number of single-family homes (standalone houses) under construction in the West region, measured in thousands of units

## United_States_Total_authorized_permits
Total number of new privately-owned housing units with authorized permits in the entire United States, measured in thousands of unit

## United_States_1Unit_authorized_permits
Number of single-family homes (standalone houses) with authorized permits in the United States, measured in thousands of units

## United_States_2to4Units_authorized_permits
Number of housing units with authorized permits that are part of small multi-family buildings, such as duplexes, triplexes, or fourplexs, measured in thousands of units

## United_States_5units_or_more_authorized_permits
Number of housing unit with authorized permits in larger apartment buildings or complexes, measured in thousands of units

## Northeast_Total_authorized_permits
Total number of new privately-owned housing units with authorized permits in the Northeast region, measured in thousands of unit

## Northeast_1Unit_authorized_permits
Number of single-family homes (standalone houses) with authorized permits in the Northeast region, measured in thousands of units

## Midwest_Total_authorized_permits
Total number of new privately-owned housing units with authorized permits in the Midwest region, measured in thousands of unit

## Midwest_1Unit_authorized_permits
Number of single-family homes (standalone houses) with authorized permits in the Midwest region, measured in thousands of units

## South_Total_authorized_permits
Total number of new privately-owned housing units with authorized permits in the South region, measured in thousands of unit

## South_1Unit_authorized_permits
Number of single-family homes (standalone houses) eith authorized permits in the South region, measured in thousands of units

## West_Total_authorized_permits
Total number of new privately-owned housing units with authorized permits in the West region, measured in thousands of unit

## West_1Unit_authorized_permits
Number of single-family homes (standalone houses) with authorized permits in the United States, measured in thousands of units

## United_States_Total_started
Total number of new privately-owned housing units started in the entire United States, measured in thousands of unit

## United_States_1Unit_started
Number of single-family homes (standalone houses) started in the United States, measured in thousands of units

## United_States_2to4Units_started
Number of housing units started that are part of small multi-family buildings, such as duplexes, triplexes, or fourplexs, measured in thousands of units

## United_States_5units_or_more_started
Number of housing unit started in larger apartment buildings or complexes, measured in thousands of units

## Northeast_Total_started
Total number of new privately-owned housing units started in the Northeast region, measured in thousands of unit

## Northeast_1Unit_started
Number of single-family homes (standalone houses) started in Northeast region, measured in thousands of units

## Midwest_Total_started
Total number of new privately-owned housing units started in the Midwest region , measured in thousands of unit

## Midwest_1Unit_started
Number of single-family homes (standalone houses) started in the Midwest region, measured in thousands of units

## South_Total_started
Total number of new privately-owned housing units started in the South Region, measured in thousands of unit

## South_1Unit_started
Number of single-family homes (standalone houses) started in the South Region, measured in thousands of units

## West_Total_started
Total number of new privately-owned housing units started in the West Region, measured in thousands of unit

## West_1Unit_started
Number of single-family homes (standalone houses) started in the West Region, measured in thousands of units

## Year
Description for  Year needs to be added

## Total
The total number of housing permits issued

## 1 Unit
Number of single-units housing permits issued

## 2 Units
Number of housing permits issued for 2 units structures

## 3 and 4 Units
Number of housing permits issued for 3-4 unit strucutres

## 5 Units or More
total number of housing units in buildings that have 5 or more units that recieved permits. For example, if there are 10 buildings with 5 units each, this column would show 50

## Num of Structures With 5 Units or More
Number of buildings with 5 or more units that received permits.

## MONTH_TOTAL_heating
A measure of how cold it was during the month (higher number, more engergy people likely used to heat their home)

## MON_DEV_FROM_NORM_heating
How much the heating degree days for the month differ from the usual amount for that month (A positive number means it was colder and a negative mean warmer)

## MON_DEV_FROM_L_YR_heating
How much the heating degree days for the month differ from the same month last year (A positive number means it was colder and a negative mean warmer)

## CUM_TOTAL_heating
The total heating degree days added up from the start of the season until now (how cold it has been overall so far this season?)

## CUM_DEV_FROM_NORM_heating
How much the total heating degree days so far differ from the usual (normal) amount for this time of year

## CUM_DEV_FROM_L_YR_heating
How much the total heating degree days so far differ from the same period last year

## CUM_DEV_FROM_NORM_PRCT_heating
The percentage difference between the total heating degree days so far and the usual (normal) amount (By what percentage has this season or year been colder or warmer tthan usual)

## CUM_DEV_FROM_L_YR_PRCT_heating
The percentage difference between the total heating degree days so far and the same period last year (By what percentage has this season or year been colder or warmer compared to last year)

## MONTH_TOTAL_cooling
A measure of how hot it was during the month (higher number, more engergy people likely used to cool their home)

## MON_DEV_FROM_NORM_cooling
How much the cooling degree days for the month differ from the usual amount for that month (A positive number means it was hotter and a negative mean colder)

## MON_DEV_FROM_L_YR_cooling
How much the cooling degree days for the month differ from the same month last year (A positive number means it was hotter and a negative mean colder)

## CUM_TOTAL_cooling
The total cooling degree days added up from the start of the season until now (how hot it has been overall so far this season?)

## CUM_DEV_FROM_NORM_cooling
How much the total cooling degree days so far differ from the usual (normal) amount for this time of year

## CUM_DEV_FROM_L_YR_cooling
How much the total cooling degree days so far differ from the same period last year

## CUM_DEV_FROM_NORM_PRCT_cooling
The percentage difference between the total cooling degree days so far and the usual (normal) amount (By what percentage has this season or year been colder or warmer than usual)

## CUM_DEV_FROM_L_YR_PRCT_cooling
The percentage difference between the total cooling degree days so far and the same period last year (By what percentage has this season or year been colder or warmer compared to last year)

## Avg_Residual
The average predictable recurring order size per loyal customer in each group. This helps normalize for territory/segment size differences when comparing performance

## Total_Residual
The sum of all predictable recurring business from loyal customers in each territory/segment/month combination. Calculated by: 1.) Identifying customers who meet loyalty criteria (≥3 active months & active in ≥25% of month consistency) 2.) Summing their average order sizes across all qualified customers in each group

## Residual_Customers
Count of unique loyal customers contributing to the residual business in each group. Indicates market penetration depth of recurring business

## Residual_Concentration
The percentage of total company-wide residual business accounted for by this specific group. Values range 0-1 where 0.01 = 1% of all residual business

