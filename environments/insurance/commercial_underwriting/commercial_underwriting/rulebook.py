"""The underwriting guidelines the expert-verified cases were decided under, verbatim from the
Snorkel Multi-Turn Insurance Underwriting benchmark (Apache-2.0), the rules they encode, and the
industry table the synthetic training cases are generated from."""

from typing import NamedTuple

GUIDELINES = """***General***
We write policies for small businesses only. Any business that fails to qualify as a "small business" is out of appetite, so we will not write a policy for it.

***Property LOB***

For property LOB, business classes pertaining to hospitality or leasing are only in appetite if the building construction is sufficiently fire resistive.
This means the building construction codes are either Type I or Type II.

Specifically, this is relevant to the following NAICS codes using the 2022 schema:

531110\tLessors of Residential Buildings and Dwellings
531120\tLessors of Nonresidential Buildings
531130\tLessors of Miniwarehouses and Self-Storage Units
531190\tLessors of Other Real Estate Property

721110\tHotels (except Casino Hotels) and Motels
721120\tCasino Hotels
721191\tBed-and-Breakfast Inns

***Policy Limits***

For most small business policies we recommend a per-occurence limit of $1 million and an aggregate limit of $2 million.

However, for cyber liability insurance coverage: Industry, business type, amount, and type of customer data handled determine how much cyber insurance businesses need.
Specifically we recommend much higher limits of $3 million per-occurence and $5 million aggregate for cyber LOB for the following NAICS classes:

NAICS code 517: Telecommunications
NAICS code 518: Computing Infrastructure Providers, Data Processing, Web Hosting, and Related Services
NAICS code 52: Finance and Insurance
NAICS code 621: Ambulatory Health Care Services
NAICS code 622310: Specialty Hospitals
NAICS code 623: Nursing and Residential Care Facilities
NAICS code 92: Public Administration

***Policy Deductibles***

For most small business policies we recommend a deductible of $500 across LOBs.

However, for auto insurance we generally recommend a $1000 deductible, and for the following NAICS codes we recommend a $5000 deductible:

NAICS code 484: Freight Trucking
NAICS code 4853: Taxi and Limousine Services
NAICS code 4854: School and Employee Bus Transportation
NAICS code 4855: Charter Bus Services
NAICS code 4871: Scenic and Sightseeing Transportation (Land)

***Auto LOB***

If our appetite guide for an auto policy indicates "qualified", then we determine final appetite according to whether the applicant has over 20 vehicles, in which case we decline because the exposure is too high.

***Worker's Compensation LOB***

If our appetite guide for a worker's compensation policy indicates "qualified", then we determine final appetite according to whether the applicant has a total annual payroll of over $5M, in which case we decline because the exposure is too high.
"""

CYBER_HIGH_LIMIT_PREFIXES = ("517", "518", "52", "621", "622310", "623", "92")
HIGH_DEDUCTIBLE_PREFIXES = ("484", "4853", "4854", "4855", "4871")
STANDARD_LIMITS = (1_000_000, 2_000_000)
CYBER_HIGH_LIMITS = (3_000_000, 5_000_000)
LOBS = ("general liability", "property", "workers compensation", "auto", "cyber", "professional liability")


def limits_for(naics: str, lob: str) -> tuple[int, int]:
    """(per-occurrence, aggregate) the guidelines recommend for a NAICS code and line of business."""
    if lob == "cyber" and naics.startswith(CYBER_HIGH_LIMIT_PREFIXES):
        return CYBER_HIGH_LIMITS
    return STANDARD_LIMITS


def deductible_for(naics: str, lob: str) -> int:
    """The deductible the guidelines recommend for a NAICS code and line of business."""
    if lob != "auto":
        return 500
    return 5000 if naics.startswith(HIGH_DEDUCTIBLE_PREFIXES) else 1000


class Industry(NamedTuple):
    naics: str
    title: str
    noun: str
    """A word for company names, e.g. `Freight` in `Blue Ridge Freight LLC`."""
    phrase: str
    """How the description introduces the business, e.g. `a charter bus operator`."""
    details: tuple[str, ...]


INDUSTRIES: tuple[Industry, ...] = (
    Industry(
        "517111",
        "Wired Telecommunications Carriers",
        "Fiber",
        "a wired telecommunications carrier",
        (
            "It operates a fiber network that delivers internet, voice and television service to residential and business subscribers.",
            "It leases fiber routes to enterprise customers and runs its own network operations center.",
        ),
    ),
    Industry(
        "517112",
        "Wireless Telecommunications Carriers (except Satellite)",
        "Wireless",
        "a regional wireless carrier",
        (
            "It runs cell towers and sells mobile voice and data plans across several counties.",
            "It resells spectrum capacity and maintains its own radio access network.",
        ),
    ),
    Industry(
        "518210",
        "Computing Infrastructure Providers, Data Processing, Web Hosting, and Related Services",
        "Cloud",
        "a web hosting and data processing provider",
        (
            "It hosts websites and databases for small businesses in its own data center and handles their backups.",
            "It processes payment and payroll batches for clients on managed servers.",
        ),
    ),
    Industry(
        "522110",
        "Commercial Banking",
        "Bank",
        "a community bank",
        (
            "It takes deposits and makes commercial and residential loans from three branches.",
            "It offers checking accounts, business lending and treasury services to local firms.",
        ),
    ),
    Industry(
        "522130",
        "Credit Unions",
        "Credit Union",
        "a member-owned credit union",
        (
            "It provides savings accounts, auto loans and mortgages to its members.",
            "It serves employees of the regional school district with deposit and loan products.",
        ),
    ),
    Industry(
        "522292",
        "Real Estate Credit",
        "Mortgage",
        "a mortgage lender",
        (
            "It originates home purchase loans and home equity lines of credit and services them in house.",
            "It underwrites residential mortgages and sells them into the secondary market.",
        ),
    ),
    Industry(
        "523150",
        "Investment Banking and Securities Intermediation",
        "Capital",
        "a securities brokerage",
        (
            "It executes trades and manages investment accounts for retail clients.",
            "It advises small companies on private placements and trades securities for customers.",
        ),
    ),
    Industry(
        "524210",
        "Insurance Agencies and Brokerages",
        "Insurance",
        "an independent insurance agency",
        (
            "It sells auto, home and commercial policies from several carriers and handles client claims paperwork.",
            "It brokers employee benefit plans and property coverage for local businesses.",
        ),
    ),
    Industry(
        "621111",
        "Offices of Physicians (except Mental Health Specialists)",
        "Medical",
        "a physician group practice",
        (
            "Its doctors see patients for primary care and minor procedures at two clinics.",
            "It runs an outpatient practice with on-site lab draws and imaging referrals.",
        ),
    ),
    Industry(
        "621210",
        "Offices of Dentists",
        "Dental",
        "a dental practice",
        (
            "Its dentists and hygienists provide cleanings, fillings and crowns.",
            "It offers general and cosmetic dentistry with an in-house lab.",
        ),
    ),
    Industry(
        "621511",
        "Medical Laboratories",
        "Diagnostics",
        "a medical testing laboratory",
        (
            "It analyzes blood and tissue samples sent by physicians and hospitals.",
            "It runs clinical chemistry and pathology testing with courier pickup from clinics.",
        ),
    ),
    Industry(
        "622310",
        "Specialty (except Psychiatric and Substance Abuse) Hospitals",
        "Hospital",
        "a specialty surgical hospital",
        (
            "It performs orthopedic and spine surgery with overnight inpatient beds.",
            "It is licensed as a hospital and specializes in cardiac care.",
        ),
    ),
    Industry(
        "623110",
        "Nursing Care Facilities (Skilled Nursing Facilities)",
        "Care",
        "a skilled nursing facility",
        (
            "It provides round-the-clock nursing care and rehabilitation to elderly residents.",
            "It operates licensed nursing beds with physical therapy and memory care units.",
        ),
    ),
    Industry(
        "623312",
        "Assisted Living Facilities for the Elderly",
        "Senior Living",
        "an assisted living community",
        (
            "Residents receive help with daily living, meals and medication management.",
            "It houses seniors in private apartments with on-site caregivers.",
        ),
    ),
    Industry(
        "484110",
        "General Freight Trucking, Local",
        "Freight",
        "a local freight trucking company",
        (
            "Its trucks haul palletized general freight within the metro area.",
            "It runs day-cab tractors delivering mixed freight to warehouses and stores.",
        ),
    ),
    Industry(
        "484121",
        "General Freight Trucking, Long-Distance, Truckload",
        "Transport",
        "a long-haul trucking carrier",
        (
            "Its tractor-trailers move full truckloads between distribution centers across the country.",
            "It hauls dry van freight on interstate routes with a fleet of sleeper cabs.",
        ),
    ),
    Industry(
        "484220",
        "Specialized Freight (except Used Goods) Trucking, Local",
        "Hauling",
        "a local specialized freight hauler",
        (
            "Its dump trucks and flatbeds move aggregate and building materials to job sites.",
            "It hauls refrigerated food products between local plants and grocers.",
        ),
    ),
    Industry(
        "485310",
        "Taxi and Ridesharing Services",
        "Cab",
        "a taxi service",
        (
            "Its drivers carry passengers on demand across the city and to the airport.",
            "It dispatches a fleet of sedans and minivans for metered rides.",
        ),
    ),
    Industry(
        "485320",
        "Limousine Service",
        "Limousine",
        "a limousine service",
        (
            "It provides chauffeured sedans and stretch limousines for weddings, corporate travel and airport runs.",
            "Its chauffeurs drive clients in luxury vehicles booked in advance.",
        ),
    ),
    Industry(
        "485410",
        "School and Employee Bus Transportation",
        "Bus",
        "a school bus contractor",
        (
            "Its buses carry students to and from public schools under district contracts.",
            "It runs employee shuttle routes for a hospital and a school district.",
        ),
    ),
    Industry(
        "485510",
        "Charter Bus Industry",
        "Coach",
        "a charter bus operator",
        (
            "It provides motorcoaches with drivers for group tours, sports teams and events.",
            "Groups hire its coaches for multi-day trips and airport transfers.",
        ),
    ),
    Industry(
        "487110",
        "Scenic and Sightseeing Transportation, Land",
        "Tours",
        "a sightseeing tour operator",
        (
            "Its open-top buses run narrated city tours for visitors.",
            "It drives tourists on guided trolley routes past historic sites.",
        ),
    ),
    Industry(
        "722511",
        "Full-Service Restaurants",
        "Bistro",
        "a full-service restaurant",
        (
            "Servers take orders at the table and the kitchen prepares dinner and weekend brunch.",
            "It seats 80 guests and serves a seasonal menu with a full bar.",
        ),
    ),
    Industry(
        "722513",
        "Limited-Service Restaurants",
        "Grill",
        "a fast-casual restaurant",
        (
            "Customers order at the counter for sandwiches, salads and bowls.",
            "It serves burgers and fries to walk-in and drive-through customers.",
        ),
    ),
    Industry(
        "445110",
        "Supermarkets and Other Grocery Retailers (except Convenience Retailers)",
        "Market",
        "a neighborhood grocery store",
        (
            "It sells fresh produce, meat, dairy and packaged goods to local shoppers.",
            "It runs a full-line supermarket with a deli and bakery counter.",
        ),
    ),
    Industry(
        "445230",
        "Fruit and Vegetable Retailers",
        "Produce",
        "a fresh produce retailer",
        (
            "It sells locally sourced fruit and vegetables from a farm-market storefront.",
            "Its stores stock seasonal produce, nuts and dried fruit.",
        ),
    ),
    Industry(
        "238220",
        "Plumbing, Heating, and Air-Conditioning Contractors",
        "Mechanical",
        "a plumbing and HVAC contractor",
        (
            "Its technicians install and repair furnaces, air conditioners and water heaters in homes.",
            "It services commercial boilers and installs ductwork on new construction.",
        ),
    ),
    Industry(
        "238210",
        "Electrical Contractors and Other Wiring Installation Contractors",
        "Electric",
        "an electrical contractor",
        (
            "Its electricians wire new homes and upgrade panels and lighting in commercial buildings.",
            "It installs and repairs electrical systems for builders and property managers.",
        ),
    ),
    Industry(
        "236115",
        "New Single-Family Housing Construction (except For-Sale Builders)",
        "Homes",
        "a custom home builder",
        (
            "It builds single-family homes on lots owned by its clients and manages the subcontractors.",
            "It constructs custom houses under contract with the homeowner.",
        ),
    ),
    Industry(
        "423730",
        "Warm Air Heating and Air-Conditioning Equipment and Supplies Merchant Wholesalers",
        "HVAC Supply",
        "an HVAC equipment distributor",
        (
            "It sells furnaces, condensers, ductwork and parts to contractors from a warehouse.",
            "Its counter and delivery trucks supply heating and cooling equipment to installers.",
        ),
    ),
    Industry(
        "532283",
        "Home Health Equipment Rental",
        "Medical Rentals",
        "a home medical equipment rental company",
        (
            "It rents hospital beds, wheelchairs and oxygen concentrators to patients at home.",
            "It delivers and services rented mobility and respiratory equipment.",
        ),
    ),
    Industry(
        "541110",
        "Offices of Lawyers",
        "Law",
        "a law firm",
        (
            "Its attorneys handle real estate closings, estate planning and small business matters.",
            "It represents clients in family law and civil litigation.",
        ),
    ),
    Industry(
        "541211",
        "Offices of Certified Public Accountants",
        "Accounting",
        "a certified public accounting firm",
        (
            "It prepares tax returns and audited financial statements for local businesses.",
            "Its CPAs provide bookkeeping, payroll and tax planning.",
        ),
    ),
    Industry(
        "541330",
        "Engineering Services",
        "Engineering",
        "a civil engineering firm",
        (
            "It designs roads, drainage and site plans for municipalities and developers.",
            "Its engineers prepare structural drawings and inspect construction.",
        ),
    ),
    Industry(
        "541511",
        "Custom Computer Programming Services",
        "Software",
        "a custom software development shop",
        (
            "It writes bespoke applications and integrations for business clients.",
            "Its developers build and maintain web applications under contract.",
        ),
    ),
    Industry(
        "541613",
        "Marketing Consulting Services",
        "Marketing",
        "a marketing consultancy",
        (
            "It plans campaigns, manages social media and runs email programs for clients.",
            "It advises companies on branding and digital advertising.",
        ),
    ),
    Industry(
        "561720",
        "Janitorial Services",
        "Cleaning",
        "a commercial cleaning company",
        (
            "Its crews clean offices, clinics and schools after hours.",
            "It provides nightly janitorial service under contracts with property managers.",
        ),
    ),
    Industry(
        "561730",
        "Landscaping Services",
        "Landscaping",
        "a landscaping company",
        (
            "Its crews mow, plant and maintain lawns and gardens for homeowners and businesses.",
            "It installs irrigation, hardscapes and seasonal plantings.",
        ),
    ),
    Industry(
        "812111",
        "Barber Shops",
        "Barbers",
        "a barber shop",
        (
            "Its barbers cut hair and trim beards for walk-in customers.",
            "It runs six chairs offering haircuts and shaves.",
        ),
    ),
    Industry(
        "812112",
        "Beauty Salons",
        "Salon",
        "a hair salon",
        (
            "Its stylists cut, color and style hair and offer manicures.",
            "It provides hair, nail and skin care services by appointment.",
        ),
    ),
    Industry(
        "811111",
        "General Automotive Repair",
        "Auto Repair",
        "an automotive repair shop",
        (
            "Its mechanics do brakes, engine diagnostics and general repairs on cars and light trucks.",
            "It services vehicles in four bays and sells replacement parts.",
        ),
    ),
    Industry(
        "458110",
        "Clothing Retailers",
        "Apparel",
        "a clothing boutique",
        (
            "It sells women's and men's apparel and accessories from a downtown storefront.",
            "It stocks casual and formal clothing and offers alterations.",
        ),
    ),
    Industry(
        "459310",
        "Florists",
        "Flowers",
        "a florist",
        (
            "It arranges bouquets for weddings, funerals and deliveries.",
            "It sells cut flowers and plants and delivers arrangements locally.",
        ),
    ),
    Industry(
        "311811",
        "Retail Bakeries",
        "Bakery",
        "a retail bakery",
        (
            "It bakes bread, cakes and pastries on site and sells them to walk-in customers.",
            "It makes custom cakes and daily breads sold from its own storefront.",
        ),
    ),
    Industry(
        "312120",
        "Breweries",
        "Brewing",
        "a craft brewery",
        (
            "It brews ale and lager sold in its taproom and to local bars.",
            "It produces beer in small batches and distributes kegs and cans regionally.",
        ),
    ),
    Industry(
        "332710",
        "Machine Shops",
        "Machining",
        "a machine shop",
        (
            "It machines metal parts to customer drawings on CNC mills and lathes.",
            "It fabricates precision components for equipment manufacturers.",
        ),
    ),
    Industry(
        "333415",
        "Air-Conditioning and Warm Air Heating Equipment and Commercial and Industrial Refrigeration Equipment Manufacturing",
        "Climate Systems",
        "a manufacturer of heating and cooling equipment",
        (
            "It builds rooftop air conditioning units and walk-in cooler systems in its plant.",
            "It manufactures commercial refrigeration equipment sold through distributors.",
        ),
    ),
    Industry(
        "339950",
        "Sign Manufacturing",
        "Signs",
        "a sign manufacturer",
        (
            "It fabricates illuminated storefront signs and vehicle graphics.",
            "It designs and builds channel letters, banners and directional signage.",
        ),
    ),
    Industry(
        "424410",
        "General Line Grocery Merchant Wholesalers",
        "Foods",
        "a grocery wholesaler",
        (
            "It distributes packaged foods and beverages to independent grocers from a warehouse.",
            "Its trucks deliver dry goods and frozen products to restaurants and stores.",
        ),
    ),
    Industry(
        "611110",
        "Elementary and Secondary Schools",
        "Academy",
        "a private K-12 school",
        (
            "It teaches kindergarten through twelfth grade on a single campus.",
            "It enrolls 300 students in elementary and secondary classes.",
        ),
    ),
    Industry(
        "611210",
        "Junior Colleges",
        "College",
        "a two-year college",
        (
            "It grants associate degrees and technical certificates on two campuses.",
            "It offers workforce training and transfer programs to about 2,000 students.",
        ),
    ),
    Industry(
        "713940",
        "Fitness and Recreational Sports Centers",
        "Fitness",
        "a fitness center",
        (
            "Members use its weight room, group classes and indoor pool.",
            "It runs a gym with personal training and swim lessons.",
        ),
    ),
    Industry(
        "721110",
        "Hotels (except Casino Hotels) and Motels",
        "Inn",
        "a hotel",
        (
            "It rents 90 guest rooms and hosts small conferences.",
            "It operates a roadside motel with a breakfast room and pool.",
        ),
    ),
    Industry(
        "721191",
        "Bed-and-Breakfast Inns",
        "Bed & Breakfast",
        "a bed-and-breakfast inn",
        (
            "It offers seven guest rooms in a historic farmhouse with breakfast included.",
            "Guests stay in a restored Victorian house and eat a home-cooked breakfast.",
        ),
    ),
    Industry(
        "531110",
        "Lessors of Residential Buildings and Dwellings",
        "Residential",
        "an apartment landlord",
        (
            "It owns and leases 120 apartment units in three buildings.",
            "It rents single-family houses and duplexes to tenants.",
        ),
    ),
    Industry(
        "531120",
        "Lessors of Nonresidential Buildings (except Miniwarehouses)",
        "Properties",
        "a commercial landlord",
        (
            "It leases office suites and retail space in two buildings it owns.",
            "It owns a strip mall and an office park rented to tenants.",
        ),
    ),
    Industry(
        "812910",
        "Pet Care (except Veterinary) Services",
        "Pet Care",
        "a pet boarding and grooming business",
        (
            "It boards dogs and cats and offers grooming and daycare.",
            "It provides dog walking, boarding and grooming services.",
        ),
    ),
    Industry(
        "541940",
        "Veterinary Services",
        "Veterinary",
        "a veterinary clinic",
        (
            "Its veterinarians treat dogs and cats and perform routine surgery.",
            "It offers vaccinations, dental cleanings and emergency care for pets.",
        ),
    ),
    Industry(
        "624410",
        "Child Care Services",
        "Kids",
        "a child day care center",
        (
            "It cares for infants and preschoolers in a licensed center on weekdays.",
            "It runs daycare and after-school programs for children.",
        ),
    ),
    Industry(
        "561612",
        "Security Guards and Patrol Services",
        "Security",
        "a security guard company",
        (
            "Its guards staff building lobbies and patrol parking lots for clients.",
            "It provides uniformed security officers for events and offices.",
        ),
    ),
)

CONSTRUCTIONS = (
    "Fire resistive (Type I)",
    "Non-combustible (Type II)",
    "Joisted masonry (Type III)",
    "Wood frame (Type V)",
)
NAME_PREFIXES = (
    "Blue Ridge",
    "Harbor",
    "Summit",
    "Prairie",
    "Lakeside",
    "Pioneer",
    "Redwood",
    "Cedar Creek",
    "Northstar",
    "Granite",
    "Riverbend",
    "Silver Oak",
    "Bayview",
    "Copper Hill",
    "Maple",
    "Evergreen",
    "Sunrise",
    "Iron Gate",
    "Clearwater",
    "Meadow",
    "Beacon",
    "Highland",
    "Willow",
    "Coastal",
    "Frontier",
    "Heritage",
    "Keystone",
    "Horizon",
)
NAME_SUFFIXES = ("LLC", "Inc.", "Co.", "Group", "Partners", "Services", "Company")
STATES = (
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
)
CITIES = (
    "Springfield",
    "Riverside",
    "Fairview",
    "Franklin",
    "Greenville",
    "Bristol",
    "Clinton",
    "Madison",
    "Georgetown",
    "Salem",
    "Ashland",
    "Burlington",
    "Dayton",
    "Lexington",
    "Milford",
    "Newport",
    "Oxford",
    "Plymouth",
    "Winchester",
    "Jackson",
    "Auburn",
    "Kingston",
    "Marion",
    "Chester",
    "Hudson",
    "Troy",
    "Dover",
    "Manchester",
    "Arlington",
    "Lancaster",
    "Columbia",
    "Bedford",
    "Newton",
    "Richmond",
    "Monroe",
    "Hamilton",
    "Lincoln",
    "Jefferson",
    "Aurora",
)
