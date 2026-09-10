"""The summarize catalog: thirteen short fictional passages (notices, memos, reports), each with the
three key points it makes and a reference summary. Fictional on purpose: the judge grades a
summary against the passage alone, so nothing here needs to be true about the world."""

from typing import TypedDict


class Entry(TypedDict):
    name: str
    passage: str
    key_points: list[str]
    reference: str


CATALOG: list[Entry] = [
    {
        "name": "library_hours",
        "passage": (
            "Starting on the first Monday of next month, the Millbrook Public Library will open at eight in the "
            "morning instead of ten on weekdays, and close at six instead of eight. Saturday hours stay the same. "
            "The change follows a survey in which most respondents said they visit before work or school. At the "
            "same time, the library is ending late fees on all books and audiobooks; items more than sixty days "
            "overdue will simply be marked lost and billed at replacement cost. Finally, the children's wing will "
            "close for renovation from the second week of the month until early spring. During the work, story "
            "time moves to the community room on the ground floor, and the children's collection will be available "
            "on request from the front desk."
        ),
        "key_points": [
            "weekday hours shift earlier, opening at eight and closing at six",
            "late fees are ending, with long-overdue items billed at replacement cost",
            "the children's wing closes for renovation until early spring",
        ],
        "reference": (
            "Millbrook Public Library is moving its weekday hours earlier (eight to six), ending late fees, and "
            "billing items over sixty days overdue at replacement cost. Its children's wing will close for "
            "renovation from the second week of next month until early spring, with story time moving to the "
            "community room."
        ),
    },
    {
        "name": "bakery_memo",
        "passage": (
            "To all staff at Harlow's Bakery: from next Tuesday we are switching our flour supplier to Northfield "
            "Mills, whose stone-ground flour absorbs more water. Expect the sourdough to need about ten percent more "
            "water and a slightly longer bulk rise; the recipe cards will be updated by Monday evening. Because the "
            "longer rise pushes the first bake later, the morning sourdough will come out at seven thirty rather "
            "than seven, and the baguettes will move to the second oven slot. Separately, the new deck oven arrives "
            "on Thursday. Marco will run a thirty-minute training on it at the start of each shift for the rest of "
            "the week, and nobody should use it before attending one. Please read the updated cards before your "
            "next shift and ask Marco if anything is unclear."
        ),
        "key_points": [
            "the flour supplier changes to Northfield Mills, so the sourdough needs more water and a longer rise",
            "the morning sourdough bake moves later, to seven thirty",
            "a new deck oven arrives Thursday and staff must attend Marco's training before using it",
        ],
        "reference": (
            "Harlow's Bakery is switching to Northfield Mills flour from Tuesday, so the sourdough recipe gets more "
            "water and a longer rise and the morning bake moves to seven thirty. A new deck oven arrives Thursday, "
            "and staff must attend one of Marco's training sessions before using it."
        ),
    },
    {
        "name": "river_cleanup",
        "passage": (
            "Saturday's cleanup along the Ashford River drew forty-two volunteers, up from twenty-six last year. "
            "Over three hours the group filled one hundred and eight bags, most of them from the stretch between "
            "the footbridge and the old mill, where the current deposits whatever washes down from town. Plastic "
            "bottles were by far the most common item, followed by food wrappers and fishing line; the group also "
            "pulled out two shopping carts and a bicycle frame. The organizers have asked the council to install "
            "three covered bins near the footbridge, which currently has none, and the council has agreed to "
            "review the request at its next meeting. The next cleanup is planned for the first weekend in October, "
            "and volunteers are asked to bring their own gloves this time because the supply ran out by nine."
        ),
        "key_points": [
            "forty-two volunteers filled one hundred and eight bags, mostly near the footbridge",
            "plastic bottles were the most common litter",
            "organizers asked the council for three covered bins near the footbridge",
        ],
        "reference": (
            "Forty-two volunteers filled one hundred and eight bags at the Ashford River cleanup, mostly plastic "
            "bottles from the stretch below the footbridge. Organizers have asked the council for three covered "
            "bins there, and the next cleanup is set for the first weekend in October."
        ),
    },
    {
        "name": "apartment_notice",
        "passage": (
            "Residents of Carver House: the water supply to the whole building will be shut off on Wednesday from "
            "nine in the morning until roughly two in the afternoon so that the main valve on the ground floor can "
            "be replaced. Please fill containers beforehand and avoid running dishwashers or washing machines that "
            "morning. On Friday the annual elevator inspection takes place; the east elevator will be out of "
            "service from eight until noon and the west elevator from noon until four, so one car will be running "
            "at all times. Lastly, the recycling company has changed its rules: glass must now go in the separate "
            "green bin by the bike shed rather than in the mixed recycling, and any mixed bag containing glass will "
            "be left uncollected. Thank you for your patience with all three of these."
        ),
        "key_points": [
            "the building's water is shut off Wednesday from nine to about two for a valve replacement",
            "the elevator inspection on Friday takes each elevator out of service for half a day, one at a time",
            "glass must now go in the separate green bin, not mixed recycling",
        ],
        "reference": (
            "Carver House will have no water on Wednesday from nine until about two while the main valve is "
            "replaced, and Friday's elevator inspection will take one elevator at a time out of service. Glass "
            "now has to go in the green bin by the bike shed instead of the mixed recycling."
        ),
    },
    {
        "name": "robotics_club",
        "passage": (
            "The Eastgate High robotics club finished third out of twenty-two teams at the regional competition "
            "last weekend, its best result so far, and qualified for the state round in March. The autonomous "
            "section scored highest; the driver-controlled section lost points when a wheel mount cracked in the "
            "final match. Repairs are underway and the team is redesigning the mount in aluminum. The club has "
            "also secured a new sponsor, Larkin Tool and Die, which is covering the state entry fee and donating "
            "machine time for the redesign. Because several members now have Thursday sports practice, weekly "
            "meetings move from Thursday to Wednesday at four in room 114, starting this week. Parents who can "
            "drive to the state round are asked to email the coach by the end of the month."
        ),
        "key_points": [
            "the club placed third of twenty-two and qualified for the state round in March",
            "Larkin Tool and Die is the new sponsor, covering the entry fee and donating machine time",
            "weekly meetings move from Thursday to Wednesday at four",
        ],
        "reference": (
            "Eastgate High's robotics club placed third of twenty-two teams at regionals and qualified for the "
            "state round in March, with new sponsor Larkin Tool and Die covering the entry fee. Meetings now take "
            "place on Wednesdays at four instead of Thursdays."
        ),
    },
    {
        "name": "ferry_schedule",
        "passage": (
            "Bayline Ferries moves to its winter timetable on the first of November. Crossings between the city "
            "pier and Gull Island drop from every thirty minutes to every hour between ten in the morning and four "
            "in the afternoon, while the early and late sailings are unchanged. Adult single fares rise by fifty "
            "cents to four dollars fifty, the first increase in three years; child and senior fares stay the same, "
            "and monthly passes bought before November keep the old price until they expire. The North Point pier "
            "will close entirely for the winter because the landing stage needs rebuilding; passengers for North "
            "Point should use the Gull Island service and the connecting shuttle bus, which will run to meet each "
            "arrival. The summer timetable and the North Point pier are expected to return in April."
        ),
        "key_points": [
            "daytime crossings drop to hourly under the winter timetable from the first of November",
            "the adult single fare rises fifty cents to four dollars fifty",
            "the North Point pier closes for the winter, replaced by the Gull Island service plus a shuttle bus",
        ],
        "reference": (
            "From the first of November Bayline Ferries runs hourly daytime crossings to Gull Island and raises "
            "the adult single fare to four dollars fifty. The North Point pier closes for the winter, with a "
            "shuttle bus from Gull Island covering it until April."
        ),
    },
    {
        "name": "garden_plots",
        "passage": (
            "The Fernside Community Garden will assign next season's plots by lottery rather than by waiting list. "
            "Anyone who submits the form by the fifteenth is entered, current plot holders included, and the draw "
            "takes place at the March meeting; households can hold at most one plot. Because last summer's water "
            "bill was nearly double the year before, watering is now limited to the hours before nine in the "
            "morning and after six in the evening, and the two taps by the shed will be locked outside those "
            "hours. To help with that, the garden is running a free compost and mulching workshop on the last "
            "Saturday of the month, led by a member who kept her beds moist through the drought with a thick straw "
            "mulch. Tools from the shed may be borrowed for the day but must be signed out in the logbook."
        ),
        "key_points": [
            "plots are assigned by lottery, with the draw at the March meeting and one plot per household",
            "watering is restricted to before nine in the morning and after six in the evening",
            "a free compost and mulching workshop runs on the last Saturday of the month",
        ],
        "reference": (
            "Fernside Community Garden will assign next season's plots by lottery at the March meeting, one per "
            "household, and now limits watering to before nine in the morning and after six in the evening. A free "
            "compost and mulching workshop takes place on the last Saturday of the month."
        ),
    },
    {
        "name": "software_release",
        "passage": (
            "Version 4.2 of the Ledgerly desktop app is available today. The headline feature is scheduled "
            "exports: reports can now be generated automatically every day, week or month and delivered to a "
            "folder or an email address, replacing the manual export most teams ran each Monday. This release "
            "also fixes the bug where imported statements with European date formats were parsed with the day and "
            "month swapped, which affected users in about a dozen countries; imports made with earlier versions "
            "are not changed retroactively and should be re-imported if the dates look wrong. The legacy CSV "
            "template introduced in version 2 is now deprecated and will be removed in version 5, planned for "
            "next spring; a converter in the settings menu migrates old templates to the current format. Updating "
            "is optional for now but will become required when version 5 ships."
        ),
        "key_points": [
            "scheduled exports can now run automatically on a daily, weekly or monthly cadence",
            "the European date-format import bug is fixed, but earlier imports must be redone by hand",
            "the version 2 CSV template is deprecated and will be removed in version 5",
        ],
        "reference": (
            "Ledgerly 4.2 adds scheduled exports and fixes the bug that swapped day and month in European date "
            "imports, though earlier imports are not corrected automatically. The version 2 CSV template is "
            "deprecated and will be removed in version 5 next spring."
        ),
    },
    {
        "name": "museum_exhibit",
        "passage": (
            "The Halden Regional Museum opens a new exhibit on the town's clockmaking trade on the ninth of "
            "October. Drawn from the workshop of the Voss family, who made tower clocks for churches across the "
            "valley for four generations, it includes the original mechanism from the town hall clock, restored to "
            "working order and running for the first time in ninety years, along with tools, order books and "
            "letters from customers. The exhibit runs until the end of February. Members are invited to a preview "
            "evening on the seventh, two days before the public opening, with a talk by the restorer at seven; "
            "places are limited to eighty and must be booked through the members' desk. School groups can book "
            "guided visits on weekday mornings, and a hands-on gear-building activity will be set up in the "
            "learning room for the whole run of the exhibit."
        ),
        "key_points": [
            "the exhibit covers the Voss family's clockmaking trade and opens on the ninth of October",
            "its centerpiece is the restored town hall clock mechanism, running for the first time in ninety years",
            "members get a preview evening on the seventh, limited to eighty booked places",
        ],
        "reference": (
            "Halden Regional Museum opens an exhibit on the Voss family's clockmaking on the ninth of October, "
            "centered on the restored town hall clock mechanism, running again after ninety years. Members can "
            "attend a preview evening on the seventh, limited to eighty booked places."
        ),
    },
    {
        "name": "farm_report",
        "passage": (
            "Hollow Creek Farm's autumn newsletter: the apple harvest came in at about two thirds of a normal year "
            "after the late frost in April took most of the early blossom, so there will be no apple boxes in the "
            "farm shop this winter and the cider run will be shorter. The squash and pumpkins, planted after the "
            "frost, did unusually well thanks to the warm, wet summer, and the shop has more than it can sell "
            "locally; the surplus is going to the food bank in Denton. Next spring the farm is planting its first "
            "acre of hazelnuts on the south slope, a long-term bet since the trees take five years to bear, and is "
            "looking for volunteers for a planting weekend in March. The Saturday market stall continues through "
            "November and then pauses until March, as usual."
        ),
        "key_points": [
            "the apple harvest was about two thirds of normal because of the April frost",
            "squash and pumpkins did very well and the surplus goes to the Denton food bank",
            "the farm will plant its first acre of hazelnuts next spring",
        ],
        "reference": (
            "Hollow Creek Farm's apple harvest was about two thirds of normal after the April frost, while squash "
            "and pumpkins did so well the surplus is going to the Denton food bank. Next spring the farm plants "
            "its first acre of hazelnuts and wants volunteers for a March planting weekend."
        ),
    },
    {
        "name": "clinic_update",
        "passage": (
            "Riverside Family Clinic is extending its flu-shot hours through October: walk-in shots are available "
            "on weekdays until seven in the evening and on Saturday mornings from eight until noon, no appointment "
            "needed, and the shot is free for patients over sixty-five and for children under five. The clinic is "
            "also switching to a new online booking system next Monday. Existing appointments carry over "
            "automatically, but patients will need to create a new login the first time they use it, and the old "
            "phone booking line will be answered only during office hours from then on. Because the pharmacy next "
            "door is expanding its building, the clinic's front parking lot will be closed for about six weeks "
            "starting mid-month; patients should use the lot behind the building, entered from Elm Street, where "
            "the clinic has reserved twenty spaces."
        ),
        "key_points": [
            "walk-in flu shots are available with extended hours through October, free for over-65s and under-5s",
            "a new online booking system starts Monday and requires a new login",
            "the front parking lot closes for about six weeks; use the lot behind the building from Elm Street",
        ],
        "reference": (
            "Riverside Family Clinic offers walk-in flu shots with extended hours through October, free for "
            "patients over sixty-five and under five, and switches to a new online booking system on Monday that "
            "needs a fresh login. Its front parking lot closes for about six weeks from mid-month, so patients "
            "should park behind the building via Elm Street."
        ),
    },
    {
        "name": "marathon_notice",
        "passage": (
            "This year's Kingsport Marathon on Sunday the twelfth follows a new route. Instead of the two laps of "
            "the harbor loop, runners will cross the river on the Union Bridge at mile eight and return over the "
            "railway bridge at mile nineteen, which organizers say removes the crowded turnaround that caused "
            "delays last year. Roads along the route close in stages from six in the morning and reopen behind "
            "the last runner, with the harbor front expected to reopen by eleven and the north bank by two in the "
            "afternoon. Residents inside the closures can drive out but not back in until their section reopens. "
            "Around three hundred volunteers are still needed for water stations and course marshalling; shifts "
            "are four hours, include a free breakfast, and can be booked on the race website until Thursday."
        ),
        "key_points": [
            "the route now crosses the river twice by bridge instead of running two harbor laps",
            "roads close in stages from six in the morning and reopen behind the last runner",
            "about three hundred volunteers are still needed, bookable until Thursday",
        ],
        "reference": (
            "The Kingsport Marathon on Sunday the twelfth switches to a route that crosses the river on two "
            "bridges instead of looping the harbor twice, with roads closing in stages from six in the morning "
            "and reopening behind the last runner. About three hundred volunteers are still needed and can sign "
            "up until Thursday."
        ),
    },
    {
        "name": "choir_season",
        "passage": (
            "The Westbrook Community Choir begins its winter season with rehearsals every Tuesday at seven thirty "
            "in St. Anne's hall, starting the first week of September. New singers are welcome without audition "
            "for the first three rehearsals, after which the director will place voices in sections; music "
            "reading is not required, as recordings of every part are shared online. The season's concert is on "
            "the fourteenth of December at the town theatre, a change from the church used in previous years, "
            "because last year's concert sold out and turned people away. Members pay a seasonal fee of forty "
            "dollars, waived for students and anyone who asks, which covers sheet music and the hall rental. The "
            "choir is also seeking a volunteer accompanist for rehearsals, as the previous pianist has moved away."
        ),
        "key_points": [
            "rehearsals run on Tuesdays at seven thirty in St. Anne's hall from early September, open to new singers",
            "the December concert moves to the town theatre because last year's sold out",
            "the choir needs a volunteer rehearsal accompanist",
        ],
        "reference": (
            "Westbrook Community Choir rehearses on Tuesdays at seven thirty in St. Anne's hall from early "
            "September, welcomes new singers without audition, and moves its December concert to the town theatre "
            "after last year's sold out. It is looking for a volunteer accompanist for rehearsals."
        ),
    },
]
