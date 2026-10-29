"""The templates of MIND core's lives (MIND's data), and every split's templates by field.

MIND's core trains on ``core-train``, locks its thresholds on ``core-val`` and is tested on
``core-test``. The three are held out from each other by **field and template**: each split has
its own fields, and each template belongs to one field. ``core-train`` takes the APD development
fields (their templates unchanged, in ``families``, but for ``CORE_KEPT_OUT``) and three fields of
its own; ``core-val`` and ``core-test`` have only their own. No core split uses an APD held-out field
or template: those stay APD's frozen test. Every field has two contradictions and one template of
each other family, as in ``families``, under the same rules (facts state no conclusion; a
resolution's numbers come from its own facts).

**A reversal's plant line** (``disambiguating``) states the raw fact that dissolves the apparent
problem, with the numbers, names and dates its planted-fact check needs, and never what it means:
never that the problem is gone (``dry since``, ``loved it``), which gave the whole item away on the
line's own day (``leak_v1.md``), nor names the apparent problem beside it. Joining it to the
earlier fact is the reader's work. Lines that broke this were rewritten (core-v2): under a new
``-v2`` id (``-v3`` when rewritten again), or under the same id in ``core-test``, whose ids the
pinned pair specs name (``pairs.pair_specs``).

**The days a fact names** hold on whichever day its schedule puts it (core-v3,
``test_apd_core_dates``): a day named relative to the writing ("Monday's rehearsal", "next week")
appears in one fact of a template at most, and every other names its date ("Sunday the 6th"),
outside the life's days, so no fact's tense turns between draws; a month named is the life's
(September 2026) or earlier. Templates rewritten for this took new ids, but in ``core-test``; APD's
own are kept out (``CORE_KEPT_OUT``), each drawn in a ``-core`` version of its own instead.

**Pair templates** (``PairTemplate``, for ``pairs``): one commitment the person relies on, in a
``core-test`` field, and its two outcomes: twice failing, or twice kept. The failed history plants
its gap (``PairTemplate.gap``); the kept one plants nothing for it. No life of a split draws them.
Pure.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import astuple, dataclass
from typing import Final

from anjo.sleep.mind.apd.families import TEMPLATES, Template
from anjo.sleep.mind.apd.families import fields as apd_fields
from anjo.sleep.mind.apd.types import CoreSplit, Family, LifeSplit


def _t(id: str, family: Family, field: str, topic: str, origins: tuple[str, ...], disambiguating: str,
       surface: str, resolution: str) -> Template:
    return Template(id, family, field, topic, origins, disambiguating, surface, resolution)


_TRUCK: Final = "food truck"
_FURNITURE: Final = "furniture making"
_PODCAST: Final = "podcast production"
_POTTERY: Final = "pottery studio"
_BIKES: Final = "bicycle shop"
_WEDDING: Final = "wedding catering"
_THEATRE: Final = "theatre production"
_PHOTO: Final = "photo exhibition"
_RENO: Final = "home renovation"  # APD development fields core-train draws templates of its own in
_FILM: Final = "independent film"
_BAKERY: Final = "small bakery"

_TRAIN: Final = (
    _t("truck-generator-load", "contradiction", _TRUCK, "the festival weekend's power", (
        "The truck's generator puts out 3,000 watts at most.",
        "For the festival weekend I'll run the fryer at 1,800 watts and the new espresso machine at 1,500 watts "
        "at the same time.",
    ), "The festival organisers rent each stall a 5,000 watt power hookup for 40 a day.",
        "The festival weekend is coming up. What should I check on the truck?",
        "The fryer and the espresso machine draw 3,300 watts against a 3,000 watt generator; rent the 5,000 watt "
        "hookup for 40 a day."),
    _t("truck-pitch-hours", "contradiction", _TRUCK, "Thursday's lunch service", (
        "The council permit lets the truck trade at the Harbour Square pitch only between 11 am and 2 pm.",
        "On Thursdays I've promised the office tower a lunch service from 1 pm to 4 pm at the pitch.",
    ), "The office tower's car park is private land, and its manager said the truck can trade there any time.",
        "Can you help me plan Thursday's lunch service for the office tower?",
        "The permit ends at 2 pm and the service runs to 4 pm; serve from the office tower's private car park."),
    _t("truck-gas-inspection", "gap", _TRUCK, "the farmers' market start", (
        "The truck starts trading at the Saturday farmers' market on the 3rd.",
        "The market requires a gas safety certificate for every truck, and the inspector books two weeks ahead.",
    ), "I haven't booked a gas inspection for the truck.",
        "Is everything ready for the farmers' market on the 3rd?",
        "The market needs a gas certificate, the inspector books two weeks ahead and none is booked; the 3rd "
        "can't hold unless an inspection is found now."),
    _t("truck-brewery-nights", "opportunity", _TRUCK, "the quiet weeknights", (
        "A craft brewery nearby has no kitchen and wants food for its quiz nights.",
        "The truck sits unused on Tuesday and Wednesday evenings.",
    ), "The brewery owner said the truck could have a free pitch in the brewery yard on quiz nights.",
        "Tuesday and Wednesday evenings are dead for me. Any ideas?",
        "Trade from the brewery yard on its quiz nights, a free pitch on the idle weeknights."),
    _t("truck-cold-reviews-v3", "reversal", _TRUCK, "the online rating", (
        "Three recent reviews say the burgers arrived cold.",
    ), "All three reviews were delivery app orders from Monday the 31st, when the app's couriers ran an hour late.",
        "Do I need a new grill because of the cold burger reviews?",
        "Not the grill: the cold burgers were delivery orders on the 31st held up by late couriers."),
    _t("truck-sunday-takings-v2", "decoy", _TRUCK, "the takings on the 6th", (
        "Takings on Sunday the 6th were 420, against a usual 900.",
        "On the 6th I sold at the same pitch with the same menu as always.",
    ), "The city marathon closed every road around the pitch until 2 pm on the 6th.",
        "The takings on the 6th were low. Should I change the menu?",
        "Not the menu: the marathon closed the roads around the pitch until 2 pm on the 6th."),
    _t("furniture-table-door", "contradiction", _FURNITURE, "the oak table delivery", (
        "The client's oak dining table is 95 centimetres wide and glued up as one solid piece.",
        "The client's flat has a front door 80 centimetres wide and no other way in.",
    ), "If the legs are bolted on instead of glued, the table top alone is 4 centimetres thick and goes through "
       "a doorway on its edge.",
        "I'm finishing the client's oak table this week. Anything to check before delivery?",
        "A 95 cm solid table won't pass an 80 cm door; bolt the legs on so the 4 cm top goes through on its edge."),
    _t("furniture-clamp-time-v2", "contradiction", _FURNITURE, "the cafe chairs", (
        "The six chairs for the cafe must be delivered on Friday the 18th at 9 am.",
        "The glued chair joints need 24 hours in clamps, I have clamps for three chairs at a time, and gluing "
        "can start on Thursday the 17th at 9 am.",
    ), "A woodworker down the road said I can borrow her clamps whenever I need them.",
        "Are the cafe chairs going to be ready for the 18th?",
        "Three chairs per 24 hours from Thursday finishes on Saturday, after Friday's 9 am delivery; borrow the "
        "woodworker's clamps and glue all six at once."),
    _t("furniture-spray-fan-v2", "gap", _FURNITURE, "the wardrobe's finish", (
        "I'm spraying lacquer on the walnut wardrobe in the workshop on the 21st.",
        "The lacquer's label says it may be sprayed only with an extraction fan running.",
    ), "The workshop has no extraction fan.",
        "I'm spraying the wardrobe on the 21st. Anything I've forgotten?",
        "The lacquer needs an extraction fan and the workshop has none; fit or rent one before spraying."),
    _t("furniture-offcut-boards", "opportunity", _FURNITURE, "the walnut offcuts", (
        "A kitchen shop in town asked whether I could supply small serving boards.",
        "I have two crates of walnut offcuts too small for furniture.",
    ), "The kitchen shop said it would take 40 boards a month at 25 each.",
        "The walnut offcuts are piling up. What would you do with them?",
        "Make the walnut offcuts into serving boards for the kitchen shop, 40 a month at 25 each."),
    _t("furniture-desk-line-v3", "reversal", _FURNITURE, "the cherry desk top", (
        "A hairline crack showed up along the cherry desk top this morning.",
    ), "The cherry desk top's wax went on in the shop at 6 degrees, and the tin warns that wax laid on below 10 "
       "degrees crazes into fine lines.",
        "Do I have to make a new top for the cherry desk?",
        "Not a cracked top: the wax went on at 6 degrees, below the 10 the tin warns of, and wax laid on that cold "
        "crazes into fine lines."),
    _t("furniture-timber-invoice", "decoy", _FURNITURE, "the timber invoice", (
        "This month's timber invoice is 1,900, against 950 last month.",
        "I'm building at the same pace as last month.",
    ), "This month's invoice includes oak for the next two commissions, bought early at a discount.",
        "The timber bill doubled. Is my supplier overcharging?",
        "Not overcharging: the invoice includes oak bought early for the next two commissions."),
    _t("podcast-edit-days-v2", "contradiction", _PODCAST, "episode 12's release", (
        "Episode 12 must be published on Monday the 21st at 6 am.",
        "The editor needs three days to edit an interview, and the episode 12 interview is recorded on Saturday "
        "the 19th.",
    ), "The editor offers a one-day rush edit for an extra 150.",
        "Can you check episode 12 is on schedule?",
        "Recording on Saturday plus three days of editing ends on Tuesday, after Monday's release; pay the 150 "
        "rush edit."),
    _t("podcast-panel-inputs", "contradiction", _PODCAST, "the live panel episode", (
        "The live panel episode has four guests, with me as host.",
        "My recorder has four microphone inputs.",
    ), "The venue's sound desk has eight inputs and can send a mix to my recorder.",
        "I'm setting up for the live panel episode. What should I check?",
        "Five people need five microphones and the recorder has four inputs; run them through the venue's "
        "eight-input desk."),
    _t("podcast-song-licence", "gap", _PODCAST, "the season trailer", (
        "The season trailer uses a 30-second clip of a chart song.",
        "Using a commercial song in a podcast needs a sync licence from its publisher, which takes about three weeks.",
    ), "I haven't contacted the song's publisher.",
        "The season trailer goes out next week. Anything left to do?",
        "The song needs a sync licence that takes three weeks and none was asked for; swap the clip or move "
        "the trailer."),
    _t("podcast-transcripts", "opportunity", _PODCAST, "the episode transcripts", (
        "A language-learning website is looking for long English conversations with transcripts.",
        "Every episode already has a full transcript that only I use, for editing.",
    ), "The website pays 60 an episode for licensed audio with a transcript.",
        "Is there anything else I can do with the transcripts I make?",
        "License the episodes with their transcripts to the language-learning website at 60 each."),
    _t("podcast-download-dip-v3", "reversal", _PODCAST, "the downloads chart", (
        "Downloads of episode 9 are down 40 percent on the dashboard against episode 8.",
    ), "The hosting service says its dashboard stopped counting one big podcast app for a week, and its "
       "corrected numbers, posted on Friday, put episode 9 at 5,200 downloads and episode 8 at 5,100.",
        "Should I change the show's format because downloads fell?",
        "Not a fall: the dashboard missed one app for a week, and the corrected downloads are level."),
    _t("podcast-sponsor-quiet", "decoy", _PODCAST, "the sponsor's renewal", (
        "The sponsor hasn't answered my renewal email for two weeks.",
        "Their marketing contact usually replies within a day.",
    ), "The sponsor's automatic reply says the marketing contact is on leave until the 20th.",
        "The sponsor has gone quiet. Are they dropping the show?",
        "Not dropping the show: the marketing contact is on leave until the 20th."),
    _t("reno-tile-shade", "reversal", _RENO, "the bathroom tiles", (
        "The bathroom wall tiles went up this morning and look two shades darker than the showroom sample.",
    ), "The tiler says new tiles and grout stay damp for about 72 hours after fixing, and damp ceramic looks darker.",
        "Should I ask the supplier to replace the bathroom tiles?",
        "Not the wrong tiles: they were held against the sample the morning they went up, and damp tiles look "
        "darker for about 72 hours."),
    # APD's templates of these fields whose facts name a day only the day they're said can place
    # (``CORE_KEPT_OUT``), each drawn by core-train in a version of its own with the day pinned.
    _t("reno-tile-membrane-core", "gap", _RENO, "the shower floor tiling", (
        "I'm tiling the shower floor on Monday the 21st.",
        "The tile adhesive needs the waterproofing membrane beneath it to cure for 48 hours first.",
    ), "No membrane has been put on the shower floor yet.",
        "I think the shower floor is ready for tiling on the 21st. Anything I'm missing?",
        "The membrane must go on and cure 48 hours before tiling and none is on; apply it now or move the tiling."),
    _t("reno-floor-cabinets-core", "contradiction", _RENO, "the kitchen work order", (
        "The kitchen floors must be refinished before the cabinets go in, so the finish isn't damaged.",
        "The cabinet delivery is booked for the 22nd and the floor refinisher can only come on the 25th.",
    ), "The cabinet company charges a fee to reschedule but allows moving a delivery up to a week later.",
        "Can you help me line up the kitchen work for the week of the 21st?",
        "The floors (25th) come after the cabinets (22nd), the wrong order; move the cabinet delivery past the 25th "
        "and pay the reschedule fee."),
    _t("film-rehearsal-late-core", "decoy", _FILM, "the lead actor's rehearsal time", (
        "The lead actor arrived at 7:40 pm to tonight's rehearsal.",
        "The call sheet said 7:00 pm for everyone.",
    ), "I texted the lead actor that she could come to rehearsals at 7:40 pm this month because of her evening "
       "class.",
        "Is the lead actor's 7:40 pm arrival at rehearsal something I should worry about?",
        "Not a problem: she was told she could come at 7:40 pm because of her evening class."),
    _t("bake-night-oven-core", "contradiction", _BAKERY, "the wholesale order on the 18th", (
        "By agreement with the neighbours, my oven can't run before 4 am.",
        "The wholesale order for Friday the 18th is 960 loaves due at the cafe by 7 am, and my oven makes 80 "
        "loaves per 25-minute cycle.",
    ), "The cafe next door has an idle oven it said I could use on Friday mornings.",
        "I want to be sure the wholesale order on the 18th goes smoothly. What should I check?",
        "960 loaves at 80 per cycle is 12 cycles, five hours; starting at 4 am ends at 9 am, too late for 7 am. "
        "Bake half in the cafe's idle oven."),
    _t("bake-online-orders-core", "decoy", _BAKERY, "the online orders on the 5th", (
        "Online orders on Saturday the 5th were 3, against a usual 15.",
        "Nothing else changed on the 5th: the same menu and the same prices.",
    ), "The ordering app emailed me that it had a maintenance outage all day on the 5th.",
        "Online orders were low on the 5th. Should I change something?",
        "Not a demand drop: the ordering app was down all day on the 5th."),
)

_VAL: Final = (
    _t("pottery-kiln-load-v2", "contradiction", _POTTERY, "the class firing", (
        "The kiln shelves hold 40 mugs per firing.",
        "The 65 mugs from the Saturday class must all go in the single firing on Wednesday the 23rd.",
    ), "The community arts centre fires its kiln on Thursdays and takes outside work.",
        "Can you help me plan the firing on the 23rd?",
        "65 mugs don't fit in 40 places; fire the other 25 in the arts centre's Thursday kiln."),
    _t("pottery-glaze-heat", "contradiction", _POTTERY, "the blue glaze order", (
        "The new blue glaze matures at 1,280 degrees.",
        "The studio kiln tops out at 1,200 degrees.",
    ), "The glaze supplier sells a version of the same blue that matures at 1,180 degrees.",
        "I'm about to order the blue glaze in bulk. Anything to check?",
        "The kiln reaches 1,200 degrees and the glaze needs 1,280; order the 1,180 degree version."),
    _t("pottery-open-day-cover", "gap", _POTTERY, "the open studio day", (
        "I'm hosting an open studio day with free wheel taster sessions on the 18th.",
        "The lease says any public event in the studio needs public liability insurance in place first.",
    ), "The studio has no public liability insurance.",
        "The open studio day is on the 18th. Is anything missing?",
        "A public event needs liability insurance under the lease and the studio has none; buy cover before "
        "the 18th."),
    _t("pottery-seconds-plates", "opportunity", _POTTERY, "the seconds shelf", (
        "A new restaurant across the street wants handmade plates with a rustic look.",
        "Forty plates with small glaze flaws sit unsold on the seconds shelf.",
    ), "The restaurant's chef said small flaws are fine and wants 30 plates by next month.",
        "The seconds shelf is full. What should I do with those plates?",
        "Sell the seconds plates to the restaurant across the street, which wants 30 and takes small flaws."),
    _t("pottery-celadon-lines-v2", "reversal", _POTTERY, "the celadon bowls", (
        "Fine cracks appeared in the glaze of the celadon bowls the day after firing.",
    ), "The celadon is a crackle glaze, sold for its web of fine lines.",
        "Should I throw away the celadon bowls?",
        "Not a defect: the celadon is a crackle glaze meant to show fine lines."),
    _t("pottery-clay-bill", "decoy", _POTTERY, "the clay delivery", (
        "The clay supplier charged 600 this month, against 300 last month.",
        "The number of students hasn't changed.",
    ), "I ordered two months of clay at once this month because the supplier closes for all of next month.",
        "The clay bill doubled. Is something off with my supplier?",
        "Not overcharging: this month's bill covers two months of clay, since the supplier closes next month."),
    _t("bikes-frame-size", "contradiction", _BIKES, "the touring bike order", (
        "The customer for the touring bike is 1.62 metres tall.",
        "The touring frame I ordered for her is a 58 centimetre frame, made for riders from 1.80 metres.",
    ), "The frame maker will swap an unbuilt frame for its 50 centimetre size, made for riders from 1.55 to "
       "1.68 metres, at no cost.",
        "The touring bike order arrives on Monday. What should I check?",
        "A 58 cm frame fits riders from 1.80 m and she is 1.62 m; swap it for the 50 cm frame at no cost."),
    _t("bikes-race-service-v2", "contradiction", _BIKES, "the club's race service", (
        "The cycling club wants its 30 bikes serviced in the week of its race on Sunday the 20th.",
        "One service takes my mechanic an hour, and he works 8 hours a day, Wednesday to Friday.",
    ), "A retired mechanic who used to work here offered to cover Saturdays this month.",
        "Can we fit the club's race service in before the race on the 20th?",
        "30 one-hour services need 30 hours and Wednesday to Friday gives 24; have the retired mechanic "
        "cover Saturday."),
    _t("bikes-battery-test", "gap", _BIKES, "the e-bike range launch", (
        "I start selling the new e-bike range on the 1st.",
        "Each e-bike battery model must pass a fire safety test before it is sold, and the lab takes four weeks.",
    ), "No battery has been sent to the lab.",
        "The e-bike range launches on the 1st. Is everything in place?",
        "The batteries need a four-week fire safety test and none was sent, so the 1st can't hold; send them "
        "now and move the launch."),
    _t("bikes-courier-fleet", "opportunity", _BIKES, "the old rental bikes", (
        "A courier company is starting a cargo service and needs sturdy bikes for its riders.",
        "Twelve old rental bikes have been stored in the back room since we stopped rentals.",
    ), "The courier company's manager said they'd buy the bikes and pay us to service them every month.",
        "What should I do with the old rental bikes in the back room?",
        "Sell the old rental bikes to the courier company and service them monthly."),
    _t("bikes-brake-squeal-v2", "reversal", _BIKES, "the new disc brakes", (
        "Customers say the new disc brakes squeal loudly on their first rides.",
    ), "The brake maker's guide says new pads squeal until they bed in over about 20 stops.",
        "Should I stop selling the new disc brakes?",
        "Not faulty: new pads squeal until they bed in, and it stops after about 20 stops."),
    _t("bikes-august-sales", "decoy", _BIKES, "August's sales", (
        "August's bike sales were 18, against 35 in July.",
        "Nothing changed in the shop's prices or stock.",
    ), "Every year since the shop opened, August sales have been about half of July's while the town is "
       "away on its summer holidays.",
        "Sales dropped in August. Do I need a promotion?",
        "Not a slump: August sales are always about half of July's while the town is away on holiday."),
)

_TEST: Final = (
    _t("wedding-oven-space", "contradiction", _WEDDING, "the Hartley wedding dinner", (
        "The Hartley wedding dinner is 180 guests, all served the main course at once.",
        "The venue kitchen's ovens hold 120 portions at a time.",
    ), "The caterer next door rents out a portable oven holding 80 portions for 90 a day.",
        "Can you help me plan the kitchen for the Hartley wedding dinner?",
        "180 portions at once against 120 oven places; rent the 80-portion portable oven for 90."),
    _t("wedding-cake-heat", "contradiction", _WEDDING, "the wedding cake", (
        "The couple chose a three-tier buttercream cake for the outdoor reception at 3 pm.",
        "The forecast for the wedding day is 31 degrees, and the baker says buttercream keeps its shape only "
        "below 25 degrees.",
    ), "The marquee company offers an air-cooled tent for the cake table.",
        "What should I think about for the cake on the day?",
        "Buttercream holds below 25 degrees and the day will be 31; keep the cake in the air-cooled tent."),
    _t("wedding-bar-notice", "gap", _WEDDING, "the evening bar", (
        "We run the evening bar at the barn wedding on the 26th.",
        "Selling drinks at a private venue needs a temporary event notice from the council at least ten working "
        "days before.",
    ), "Nobody has filed a temporary event notice for the barn.",
        "The barn wedding is on the 26th. Is the bar all sorted?",
        "The bar needs a temporary event notice ten working days ahead and none is filed; file it now or the "
        "bar can't open on the 26th."),
    _t("wedding-office-lunches", "opportunity", _WEDDING, "the idle weekdays", (
        "The wedding planner's office asked me whether I know a weekday lunch caterer.",
        "My kitchen and two cooks sit idle Monday to Thursday.",
    ), "The planner's office wants lunch for 15 people four days a week and pays 12 a head.",
        "My kitchen is quiet early in the week. Any ideas?",
        "Cater the planner's office lunches on the idle weekdays, 15 people four days a week at 12 a head."),
    _t("wedding-salty-lamb", "reversal", _WEDDING, "the tasting feedback", (
        "The couple at the tasting on Saturday the 5th said the lamb was too salty.",
    ), "The tasting lamb was the only dish a trainee cooked, while the head chef was away on the 5th.",
        "Should I take the lamb off the wedding menu?",
        "Not the recipe: a trainee cooked the salty tasting batch while the head chef was away."),
    _t("wedding-rsvp-count", "decoy", _WEDDING, "the guest numbers", (
        "Only 90 of the 150 invited guests have replied to the invitation.",
        "The final numbers are due to me in a week.",
    ), "The couple told me the other 60 are family who confirmed by phone and will be on the final list.",
        "Only 90 people have replied. Should I cut the food order?",
        "Not a shortfall: the other 60 confirmed by phone and will be on the final list."),
    _t("theatre-wall-height", "contradiction", _THEATRE, "the touring set", (
        "The castle wall in our set stands 4.5 metres tall.",
        "The second tour venue's stage has 4 metres of clearance below its lighting grid.",
    ), "The wall is built in three 1.5 metre sections that can stand two high.",
        "We load the set into the second tour venue next week. Anything to check?",
        "A 4.5 m wall won't fit under 4 m of clearance; stand two 1.5 m sections for 3 m at that venue."),
    _t("theatre-tech-cast", "contradiction", _THEATRE, "the tech rehearsal", (
        "The tech rehearsal needs the full cast of 14 on stage on the 17th.",
        "Four of the cast are in another show's matinee on the 17th until 5 pm.",
    ), "The theatre is free on the evening of the 17th from 6 pm.",
        "Can you look over the plan for the tech rehearsal on the 17th?",
        "Four actors are in a matinee until 5 pm on the 17th; start the tech rehearsal at 6 pm, when the "
        "theatre is free."),
    _t("theatre-candle-approval", "gap", _THEATRE, "the opening night", (
        "Opening night is on the 19th, and the set has real candles on stage.",
        "Any naked flame on stage needs the fire officer's written approval before the first performance.",
    ), "Nobody has asked the fire officer about the candles.",
        "Opening night is on the 19th. Is anything still missing?",
        "Naked flame needs the fire officer's written approval before the first show and nobody asked; ask now "
        "or use electric candles."),
    _t("theatre-school-matinee", "opportunity", _THEATRE, "the empty Wednesday afternoons", (
        "The local secondary school is studying this play for its exams.",
        "The theatre is dark on Wednesday afternoons during our run.",
    ), "The school's head of English said they'd book a full house of 200 students for a Wednesday matinee "
       "at 8 each.",
        "Wednesday afternoons are dead. Any way to use them?",
        "Run a Wednesday matinee for the school studying the play: a full house of 200 at 8 each."),
    _t("theatre-hoarse-lead", "reversal", _THEATRE, "the lead's voice", (
        "The lead actor's voice was hoarse at the rehearsal on Sunday the 6th.",
    ), "The lead had shouted through a football match on Saturday the 5th, and he is resting his voice ahead of "
       "the run on Thursday the 17th.",
        "Do I need to bring in the understudy for the lead?",
        "Not needed: the lead was hoarse from shouting at the match on the 5th, and he is resting his voice before "
        "the run on the 17th."),
    _t("theatre-advance-sales", "decoy", _THEATRE, "the advance ticket sales", (
        "Advance sales for the second week are 120 tickets, against 400 for the first week.",
        "The marketing hasn't changed between the two weeks.",
    ), "The second week's tickets went on sale on Tuesday the 1st, and the first week's had been on sale "
       "since early July.",
        "Second week sales look weak. Should I cut prices?",
        "Not weak: the second week's tickets went on sale on the 1st, the first week's back in early July."),
    _t("photo-wall-length", "contradiction", _PHOTO, "the gallery hang", (
        "The gallery's walls give 30 metres of hanging space.",
        "The 24 framed prints need 1.5 metres of wall each, spacing included.",
    ), "The gallery has two freestanding panels, 4 metres each, that can stand in the middle of the room.",
        "I'm planning the hang at the gallery. What should I check?",
        "24 prints at 1.5 m need 36 m against 30 m of wall; add the two 4 m freestanding panels."),
    _t("photo-print-width", "contradiction", _PHOTO, "the large prints", (
        "The three largest prints in the show must be 2 metres wide.",
        "My usual print lab prints up to 1.5 metres wide.",
    ), "A lab in the next city prints up to 3 metres wide and delivers in four days.",
        "I'm sending the large prints off this week. Anything to check?",
        "2 m prints are wider than the usual lab's 1.5 m; send them to the lab in the next city, which "
        "prints to 3 m."),
    _t("photo-model-releases", "gap", _PHOTO, "the portrait series", (
        "Eight prints in the show are street portraits with the people's faces clearly visible.",
        "The gallery requires a signed model release for every recognisable person before the opening.",
    ), "I haven't collected any model releases.",
        "The opening is next Friday. Is the portrait series ready?",
        "The gallery needs signed releases for the eight portraits and none are collected; get them or pull "
        "those prints before the opening."),
    _t("photo-hotel-rooms", "opportunity", _PHOTO, "the stored prints", (
        "A boutique hotel opening downtown is looking for local photography for its rooms.",
        "I have 40 unsold prints from my last show in storage.",
    ), "The hotel's designer said they'd buy 30 framed prints at 150 each.",
        "What can I do with the prints sitting in storage?",
        "Sell 30 of the stored prints to the boutique hotel at 150 each."),
    _t("photo-green-cast", "reversal", _PHOTO, "the test prints", (
        "The test prints came back with a strong green cast.",
    ), "I'd viewed the test prints under the studio's new fluorescent tubes, which give off a green tint.",
        "Do I need to reprint everything because of the green cast?",
        "Not the prints: the green came from the studio's new fluorescent tubes they were viewed under."),
    _t("photo-framing-quote", "decoy", _PHOTO, "the framing quote", (
        "The framer quoted 3,600 for this show, against 1,800 for the last one.",
        "Both shows have the same number of prints.",
    ), "This show's prints are twice the size and use museum glass, which the last show's didn't.",
        "Is the framer overcharging me this time?",
        "Not overcharging: this show's frames are twice the size, with museum glass."),
)

@dataclass(frozen=True)
class PairTemplate:
    """A commitment stated on day 1 of both histories; ``failed`` or ``kept``, its two outcomes on
    the fork's days; ``ask``, the last day's question, the same in both. The failed history's gap
    is planted from ``topic`` and ``resolution``."""

    id: str
    field: str
    topic: str
    commitment: str
    failed: tuple[str, str]
    kept: tuple[str, str]
    ask: str
    resolution: str

    @property
    def gap(self) -> Template:
        """The failed history's planted item: the commitment and the first failure are its origin,
        the second failure settles it."""
        return Template(f"{self.id}-failed", "gap", self.field, self.topic, (self.commitment, self.failed[0]),
                        self.failed[1], self.ask, self.resolution)


def _p(id: str, field: str, topic: str, commitment: str, failed: tuple[str, str], kept: tuple[str, str], ask: str,
       resolution: str) -> PairTemplate:
    return PairTemplate(id, field, topic, commitment, failed, kept, ask, resolution)


PAIR_TEMPLATES: Final = (
    _p("pair-fish-supplier", _WEDDING, "the Ashford wedding's fish",
       "Harbour Fresh agreed to deliver 40 kilograms of sea bass for the Ashford wedding on the 19th.",
       ("Harbour Fresh cancelled the fish delivery for the Morton party two hours before it was due.",
        "Harbour Fresh cancelled again: the fish order for the Lee birthday never arrived."),
       ("Harbour Fresh delivered the fish for the Morton party on time.",
        "Harbour Fresh's fish order for the Lee birthday arrived early and complete."),
       "The Ashford wedding is on the 19th. What should I get done this week?",
       "Harbour Fresh has cancelled twice since agreeing to supply the Ashford wedding on the 19th; line up a "
       "backup fish supplier now."),
    _p("pair-cold-van", _WEDDING, "the food transport for the Pemberton wedding",
       "Coldline is renting me a refrigerated van to take the food to the Pemberton wedding on the 19th.",
       ("Coldline's van broke down on the way to the Morton party, and the food sat in the heat for an hour.",
        "Coldline's van wouldn't start for this morning's lunch run, so I used taxis."),
       ("Coldline's van kept the Morton party food cold the whole way.",
        "Coldline's van ran this morning's lunch run without a hitch."),
       "The Pemberton wedding is on the 19th. What's left for me to sort?",
       "Coldline's van has failed twice since it was booked for the Pemberton wedding on the 19th; book a "
       "second refrigerated van as a backup."),
    _p("pair-lighting-desk", _THEATRE, "the lighting for the opening",
       "Brightline is supplying the lighting desk for our opening on the 19th.",
       ("The lighting desk Brightline lent us for rehearsals crashed in the middle of today's run.",
        "Brightline's desk froze again tonight and lost every lighting cue."),
       ("The lighting desk Brightline lent us for rehearsals ran the whole of today's run.",
        "Brightline's desk ran tonight's rehearsal and kept every lighting cue."),
       "We open on the 19th. What should I check this week?",
       "Brightline's desk crashed twice in rehearsals before the opening on the 19th; arrange a backup desk or "
       "a technician on site."),
    _p("pair-programme-printer", _THEATRE, "the opening night programmes",
       "Inkwell is printing 500 programmes for the opening on the 19th.",
       ("Inkwell delivered our poster run three days after the agreed date.",
        "Inkwell's flyers came back late as well, and with the wrong venue address."),
       ("Inkwell delivered our poster run on the agreed date.",
        "Inkwell's flyers came back on time and spotless."),
       "The opening is on the 19th. Is there anything I should chase?",
       "Inkwell has been late twice since taking the programme order for the 19th; confirm their date now and "
       "line up a backup printer."),
    _p("pair-framer", _PHOTO, "the frames for the opening",
       "Oakframe is framing the 16 prints for the opening on the 19th.",
       ("Oakframe sent back the first frames four days late, and two of them were cracked.",
        "Oakframe pushed the second batch of frames back another week."),
       ("Oakframe sent back the first frames on the agreed day, every one perfect.",
        "Oakframe's second batch of frames arrived two days early."),
       "The opening is on the 19th. What should I be doing this week?",
       "Oakframe has slipped twice on the frames for the opening on the 19th; confirm the remaining frames now "
       "and line up a backup framer."),
    _p("pair-projector", _PHOTO, "the slideshow at the opening",
       "Lumen is lending us a projector for the slideshow at the opening on the 19th.",
       ("The Lumen projector we tested overheated and shut off after 20 minutes.",
        "The replacement Lumen projector shut off after 25 minutes as well."),
       ("The Lumen projector we tested ran for three hours without trouble.",
        "The Lumen projector ran the whole slideshow again today without trouble."),
       "The opening is on the 19th. Anything I should check this week?",
       "Lumen's projectors shut off twice in testing before the opening on the 19th; arrange a backup projector."),
)

CORE_TEMPLATES: Final[tuple[Template, ...]] = (*_TRAIN, *_VAL, *_TEST)
CORE_FIELDS: Final[dict[CoreSplit, tuple[str, ...]]] = {
    "core-train": (*apd_fields("development"), _TRUCK, _FURNITURE, _PODCAST),
    "core-val": (_POTTERY, _BIKES),
    "core-test": (_WEDDING, _THEATRE, _PHOTO),
}
_ALL: Final[tuple[Template, ...]] = (*TEMPLATES["development"], *TEMPLATES["held_out"], *CORE_TEMPLATES)
# APD's development templates core-train doesn't draw; APD's templates stay frozen, so core-train
# draws its own in their place (``_TRAIN``): a reversal whose plant line states its outcome
# (``reno-tile-shade`` in its place), and those whose facts name a day only the day they're said can
# place, or a date the life's days fall on both sides of (each ``-core`` in its place).
CORE_KEPT_OUT: Final = frozenset({
    "reno-damp-patch", "reno-tile-membrane", "reno-floor-cabinets", "film-rehearsal-late", "bake-night-oven",
    "bake-online-orders",
})
_CORE_POOL: Final[tuple[Template, ...]] = tuple(t for t in _ALL if t.id not in CORE_KEPT_OUT)
_PAIR_GAPS: Final = tuple(pair.gap for pair in PAIR_TEMPLATES)


def fields(split: LifeSplit) -> tuple[str, ...]:
    """The split's fields, in order: APD's for its splits (``families``), the core's for its own."""
    if split == "development" or split == "held_out":
        return apd_fields(split)
    return CORE_FIELDS[split]


def templates_for(split: LifeSplit, field: str) -> tuple[Template, ...]:
    """The split's templates in ``field``; empty for a field the split doesn't hold."""
    if field not in fields(split):
        return ()
    pool = TEMPLATES[split] if split == "development" or split == "held_out" else _CORE_POOL
    return tuple(template for template in pool if template.field == field)


TEXTS_DIGEST_CHARS: Final = 16


def texts_digest(split: LifeSplit, field: str) -> str:
    """A short digest of every template text a life or pair of ``split`` in ``field`` can draw: the
    field's draw pool and its pair templates. A life's and a pair's made record keep it, so lives
    written from other texts are never mixed (``gen_batch.refuse_changed``). The field's texts, not
    the life's own draw: a digest of what a sealed life drew would name its planted templates to
    anyone who tried the field's few combinations."""
    drawn = sorted(templates_for(split, field), key=lambda t: t.id)
    paired = sorted((pair for pair in PAIR_TEMPLATES if pair.field == field), key=lambda pair: pair.id)
    texts = json.dumps([*(astuple(t) for t in drawn), *(astuple(pair) for pair in paired)], ensure_ascii=False)
    return hashlib.sha256(texts.encode()).hexdigest()[:TEXTS_DIGEST_CHARS]


def template(template_id: str) -> Template | None:
    """The template of any split, or a pair's gap, with that id."""
    return next((t for t in (*_ALL, *_PAIR_GAPS) if t.id == template_id), None)
