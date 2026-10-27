"""The five problem families and their templates (Task 17).

A template is one planted problem written out: the facts the person states in passing
(``origins``, then ``disambiguating``), what she says when she raises it (``surface``), the neutral
``topic`` the discoverability reader gets, and the sealed ``resolution``. Facts are plain
statements; the life writer words them in the person's voice and never joins them.

The two splits share the families and nothing else: each split has its own fields, and each
template belongs to one field, so templates, fields and every sentence below are disjoint between
the splits. Every field has two contradictions and one template of each other family, so a life of
up to ``MAX_PREVALENCE`` items can be drawn from one field. Pure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from anjo.sleep.mind.apd.types import Family, Split
from anjo.sleep.mind.inquiry._check import check_text


@dataclass(frozen=True)
class Template:
    """One planted problem. A reversal has one origin fact; every other family has two."""

    id: str
    family: Family
    field: str
    topic: str
    origins: tuple[str, ...]
    disambiguating: str
    surface: str
    resolution: str

    def __post_init__(self) -> None:
        for name in ("id", "field", "topic", "disambiguating", "surface", "resolution"):
            check_text(name, getattr(self, name))
        if len(self.origins) != (1 if self.family == "reversal" else 2):
            raise ValueError("a reversal has one origin fact and every other family two")
        for fact in self.origins:
            check_text("origins", fact)
        if self.topic in (self.id, self.resolution):
            raise ValueError("a topic is neutral: neither the template id nor the resolution")


def _t(id: str, family: Family, field: str, topic: str, origins: tuple[str, ...], disambiguating: str,
       surface: str, resolution: str) -> Template:
    return Template(id, family, field, topic, origins, disambiguating, surface, resolution)


_FILM: Final = "independent film"
_RENO: Final = "home renovation"
_BAKERY: Final = "small bakery"
_BOARD: Final = "board game publishing"
_GARDEN: Final = "urban gardening"
_MUSIC: Final = "indie music release"

_DEVELOPMENT: Final = (
    _t("film-lens-room", "contradiction", _FILM, "the scene 4 shoot", (
        "Scene 4 will be shot on an 85 mm lens at full frame, as a medium shot, for the compressed look.",
        "Scene 4 takes place in a kitchen that is 3 metres wide, and the camera operator works handheld "
        "with no room to back up.",
    ), "The hallway beside the kitchen is 4 metres deep and opens into the kitchen through a doorway.",
        "I need to lock the plan for the scene 4 shoot this week. What should I check?",
        "An 85 mm medium shot needs about 3.5 metres of distance and the kitchen is 3 metres wide; "
        "shoot scene 4 from the hallway through the doorway."),
    _t("film-crew-cap", "contradiction", _FILM, "the scene 7 night shoot", (
        "Payroll covers exactly five people on set, including me as director.",
        "For the night exterior in scene 7 I booked a camera operator, a gaffer, two grips and a sound mixer.",
    ), "My cousin offered to work any weekend shoot for free, and scene 7 is shot on a Saturday.",
        "Can you help me get ready for the scene 7 night shoot?",
        "Six people are booked against five paid places; the unpaid cousin on Saturday fills the sixth, "
        "so scene 7 works only if the cousin joins."),
    _t("film-street-permit", "gap", _FILM, "the chase scene on Alder Street", (
        "Scene 9 is a chase down Alder Street, and I'm shooting it on the 20th.",
        "The city film office needs a street permit application ten working days before any shoot "
        "that closes a street.",
    ), "I haven't filed anything with the city film office for any scene.",
        "I want to confirm everything for the Alder Street chase. Where do we stand?",
        "The chase closes a street, so a permit must be filed ten working days ahead; none was filed, "
        "so the 20th must move or a rush exception be requested now."),
    _t("film-score-room", "opportunity", _FILM, "the score recording", (
        "A violinist friend is looking for a quiet room to record a short piece for free.",
        "The church hall I booked for scene 12 stands empty every evening after the shoot wraps at six.",
    ), "The church hall caretaker said any recording is fine as long as it ends by ten.",
        "I'm trying to sort out the music for the film. Any ideas?",
        "The violinist can record the piece in the booked church hall after the shoot wraps, "
        "any evening before ten, for free."),
    _t("film-camera-match", "reversal", _FILM, "the two camera bodies", (
        "The footage from our second camera body looks green next to the footage from the first.",
    ), "The colorist said the second body was set to a different white balance preset, and matching "
       "the setting makes the two identical.",
        "Should I rent a third camera because the two we have don't match?",
        "Not a problem: the green look was a white balance preset, fixed by matching the setting."),
    _t("film-rehearsal-late", "decoy", _FILM, "the lead actor's rehearsal time", (
        "The lead actor arrived at 7:40 pm to Tuesday's rehearsal.",
        "The call sheet said 7:00 pm for everyone.",
    ), "I texted the lead actor that she could come at 7:40 pm on Tuesdays because of her evening class.",
        "Is the lead actor's arrival on Tuesday something I should worry about?",
        "Not a problem: she was told 7:40 pm was fine on Tuesdays."),
    _t("reno-island-width", "contradiction", _RENO, "the kitchen island", (
        "The kitchen island will be 1.6 metres deep.",
        "The kitchen is 3.4 metres wide, and building code needs a clear path of one metre on each side "
        "of an island.",
    ), "The wall on the west side of the kitchen can be removed, opening onto the dining room.",
        "I'm ordering the kitchen island this week. What should I double-check?",
        "A 1.6 metre island plus two one-metre paths needs 3.6 metres against 3.4; slim the island to "
        "1.4 metres or open the west wall into the dining room."),
    _t("reno-floor-cabinets", "contradiction", _RENO, "the kitchen delivery schedule", (
        "The floors must be refinished before the cabinets go in, so the finish isn't damaged.",
        "The cabinet delivery is booked for the 12th and the floor refinisher can only come on the 15th.",
    ), "The cabinet company charges a fee to reschedule but allows moving a delivery up to a week later.",
        "Can you help me line up the kitchen work for next week?",
        "The floors (15th) come after the cabinets (12th), the wrong order; move the cabinet delivery "
        "past the 15th and pay the reschedule fee."),
    _t("reno-tile-membrane", "gap", _RENO, "the bathroom tiling", (
        "I'm tiling the bathroom wall next Monday.",
        "The tile adhesive needs the waterproofing membrane beneath it to cure for 48 hours first.",
    ), "No membrane has been put on the bathroom wall yet.",
        "I think the bathroom is ready for tiling on Monday. Anything I'm missing?",
        "The membrane must go on and cure 48 hours before tiling and none is on; apply it now "
        "or move the tiling."),
    _t("reno-skip-share", "opportunity", _RENO, "the demolition rubble", (
        "The neighbour is renting a skip bin for the week of the 8th.",
        "My demolition will produce about a tonne of rubble that I planned to haul to the tip on weekends.",
    ), "The neighbour said they'd happily share the bin; it's a large six-metre one, half empty.",
        "How should I deal with the rubble from the demolition?",
        "Share the neighbour's skip bin for the rubble instead of hauling it to the tip."),
    _t("reno-damp-patch", "reversal", _RENO, "the hallway ceiling", (
        "A damp patch appeared on the hallway ceiling after Tuesday's rain.",
    ), "The plumber found it was condensation from an uninsulated pipe, which he has since lagged, "
       "and the patch dried within two days.",
        "Do I need a roofer because of the damp patch in the hallway?",
        "Not a leak: condensation from an uninsulated pipe, already fixed by lagging it."),
    _t("reno-wiring-quote", "decoy", _RENO, "the electrician's quote", (
        "The electrician quoted 2,400 to rewire the ground floor.",
        "A friend's rewiring of a similar house last year cost 1,500.",
    ), "The quote includes replacing the old fuse box and adding a garden socket circuit, which my "
       "friend's job did not.",
        "Is the electrician's quote too high?",
        "Not overpriced: the quote covers a fuse box and a garden circuit the friend's job lacked."),
    _t("bake-night-oven", "contradiction", _BAKERY, "Friday's wholesale order", (
        "By agreement with the neighbours, the oven can't run before 4 am.",
        "Friday's order is 960 loaves due at the cafe by 7 am, and my oven makes 80 loaves per "
        "25-minute cycle.",
    ), "The cafe next door has an idle oven it said I could use on Friday mornings.",
        "I want to be sure Friday's wholesale order goes smoothly. What should I check?",
        "960 loaves at 80 per cycle is 12 cycles, five hours; starting at 4 am ends at 9 am, too late "
        "for 7 am. Bake half in the cafe's idle oven."),
    _t("bake-flour-discount", "contradiction", _BAKERY, "the flour order", (
        "The flour supplier gives the bulk discount only on orders of 500 kg or more.",
        "The storage room holds 300 kg of flour at most.",
    ), "The supplier will split a 500 kg order into two deliveries two weeks apart at the same discount.",
        "I'm about to place the flour order. Anything to think about?",
        "A 500 kg order doesn't fit in 300 kg of storage; ask for the split delivery at the same discount."),
    _t("bake-registration", "gap", _BAKERY, "the second shop's opening", (
        "The second shop opens on the 1st with a menu of 20 pastries.",
        "Selling packaged food needs a food registration from the council, which takes five weeks to process.",
    ), "I haven't applied to the council for the second shop.",
        "Everything for the second shop's opening seems on track. Right?",
        "The shop needs a registration that takes five weeks and none was applied for, so the 1st "
        "can't hold; apply now and move the opening."),
    _t("bake-bookshop-corner", "opportunity", _BAKERY, "the surplus croissants", (
        "A local bookshop wants a cafe corner with pastries but has no kitchen.",
        "I have surplus croissants most Sundays that go in the bin.",
    ), "The bookshop said it would pay a fixed price per tray and has a fridge to store them.",
        "What should I do about the croissants left over on Sundays?",
        "Sell the Sunday surplus croissants to the bookshop for its cafe corner."),
    _t("bake-rye-sales", "reversal", _BAKERY, "the rye loaf's sales", (
        "Sales of the rye loaf fell by half over the last two weeks.",
    ), "The flour mill's deliveries stopped for two weeks because of road works, so the rye was out of "
       "stock, and sales recovered once it returned.",
        "Should I drop the rye loaf because it stopped selling?",
        "Not falling demand: the loaf was out of stock for two weeks and sold normally on its return."),
    _t("bake-online-orders", "decoy", _BAKERY, "Monday's online orders", (
        "Online orders on Monday were 3, against a usual 15.",
        "Nothing else changed on Monday: the same menu and the same prices.",
    ), "The ordering app emailed me that it had a maintenance outage all of Monday.",
        "Online orders were low on Monday. Should I change something?",
        "Not a demand drop: the ordering app was down all Monday."),
)

_HELD_OUT: Final = (
    _t("board-box-size", "contradiction", _BOARD, "the game box", (
        "The printer's standard box fits a board up to 40 centimetres square.",
        "The board art is final at 45 by 45 centimetres, folded once.",
    ), "The printer offers a 48 centimetre box for 1.20 more per unit, and I have 2.00 of margin per unit.",
        "I'm sending the game to the printer this week. What should I check first?",
        "A 45 cm board doesn't fit the 40 cm box; order the 48 cm box, which the margin covers."),
    _t("board-delivery-date", "contradiction", _BOARD, "the backer delivery date", (
        "The campaign page promises backers their copies in March.",
        "The factory needs 14 weeks from final files, and the files won't be final before mid-January.",
    ), "The 14 weeks include six weeks by sea; air freight would cut the total to nine weeks.",
        "Backers are asking when they'll get their copies. What do I tell them?",
        "Mid-January plus 14 weeks lands in late April; air freight brings it to about 20 March, "
        "the only way to keep the promise."),
    _t("board-safety-test", "gap", _BOARD, "the German shop launch", (
        "I start selling the game to shops in Germany on the 1st of next month.",
        "Selling a game in the EU needs a safety test report, and the test lab takes about six weeks.",
    ), "I haven't commissioned any safety test for the game.",
        "The shop launch in Germany is set for the 1st. Is there anything I should prepare?",
        "The safety test report takes six weeks and none was commissioned, so the 1st cannot hold; "
        "commission it now and move the launch."),
    _t("board-overstock-cafe", "opportunity", _BOARD, "the first edition stock", (
        "A card game cafe owner wants exclusive games to sell.",
        "Three hundred copies of my first edition are sitting in my garage.",
    ), "The cafe owner confirmed they'd buy first-edition copies at the wholesale price in lots of 50.",
        "What can I do with all the first edition copies?",
        "Sell the first-edition copies to the cafe in lots of 50 at wholesale."),
    _t("board-dice-reviews", "reversal", _BOARD, "the shop site reviews", (
        "Reviews on the shop site say the dice are too small.",
    ), "The reviews are all about the prototype set, which used old dice; production dice are twice "
       "the size.",
        "Do I need to redesign the dice because of the reviews?",
        "Not a problem: the reviews describe the prototype's old dice, and the production dice are larger."),
    _t("board-print-quote", "decoy", _BOARD, "the print run quote", (
        "The print run quote is 6,800, double last year's 3,400.",
        "Last year's run was 1,000 copies.",
    ), "This year's run is 2,000 copies with a linen finish added.",
        "Is the printer's quote this year out of line?",
        "Not out of line: twice the copies with a finish added makes the per-copy price comparable."),
    _t("garden-railing-load", "contradiction", _GARDEN, "the balcony planters", (
        "The balcony railing is rated to carry 150 kilograms in total.",
        "Three planters of about 60 kilograms each when wet will hang from the railing.",
    ), "The balcony floor is a concrete slab rated far above that, so planters can stand on it.",
        "I'm hanging planters on the balcony this weekend. Anything to check?",
        "Three 60 kg planters weigh 180 kg against a 150 kg railing; stand them on the slab instead."),
    _t("garden-frost-date", "contradiction", _GARDEN, "the seedlings' move outdoors", (
        "The seedlings must go outdoors by the 10th or they get leggy.",
        "The last frost here is usually around the 25th.",
    ), "A cold frame in the community garden is free for members, and I'm a member.",
        "Is the 10th still a good day to put the seedlings outside?",
        "Outdoors on the 10th risks frost until the 25th; harden them in the community cold frame."),
    _t("garden-hive-register", "gap", _GARDEN, "the rooftop hives", (
        "I'm putting two beehives on the roof in April.",
        "The city requires hives to be registered before they are installed.",
    ), "I haven't registered anything with the city.",
        "The rooftop hives are all set for April. Anything left?",
        "Hives must be registered before installing and nothing is registered; register first or "
        "the April installation is not allowed."),
    _t("garden-coffee-grounds", "opportunity", _GARDEN, "the compost heap", (
        "The cafe downstairs throws away about 5 kilograms of coffee grounds a day.",
        "My compost heap is dry and short of nitrogen.",
    ), "The cafe owner said I can take the grounds any morning.",
        "My compost heap isn't doing well. What would you try?",
        "Collect the cafe's coffee grounds each morning to feed the heap's missing nitrogen."),
    _t("garden-black-basil", "reversal", _GARDEN, "the basil plants", (
        "The basil leaves turned black overnight.",
    ), "It was one cold night at 3 degrees; the nursery said the plants recover from the base, and new "
       "shoots appeared a week later.",
        "Should I pull the basil out and start again?",
        "Not dead: one cold night blackened the leaves, and the plants regrew from the base."),
    _t("garden-water-bill", "decoy", _GARDEN, "this month's water bill", (
        "My water bill this month is double last month's.",
        "I water the plants the same way as before.",
    ), "The utility skipped last month's meter reading, so this bill covers two months.",
        "My water bill doubled. Is my irrigation leaking?",
        "Not a leak: the bill covers two months because a reading was skipped."),
    _t("music-mastering-date", "contradiction", _MUSIC, "the album mastering", (
        "The album must be mastered by the 5th to reach the pressing plant on time.",
        "The mastering engineer is on holiday until the 12th.",
    ), "A second engineer at the same studio mastered two of my earlier singles and has openings on the 3rd.",
        "Is the album on track for the pressing deadline?",
        "The engineer is away until the 12th, past the 5th; book the second engineer on the 3rd."),
    _t("music-pressing-budget", "contradiction", _MUSIC, "the vinyl pressing", (
        "The pressing plant's minimum order is 300 records at 4 each.",
        "My budget for pressing is 800 in total.",
    ), "The plant offers a short run of 100 records at 7 each.",
        "I'm about to order the vinyl. How should I go about it?",
        "The minimum run costs 1,200 against 800; the 100-record short run costs 700 and fits."),
    _t("music-artwork-upload", "gap", _MUSIC, "the single's release", (
        "The single comes out on the 14th on all streaming services.",
        "The distributor needs the track and the artwork uploaded two weeks before release.",
    ), "The artwork isn't done; the designer hasn't started.",
        "Is the single ready to come out on the 14th?",
        "The artwork is not done and both files are due two weeks ahead, so the 14th cannot hold "
        "unless the artwork lands now."),
    _t("music-podcast-theme", "opportunity", _MUSIC, "the unreleased instrumental", (
        "A podcast host asked me for theme music.",
        "I have an unreleased instrumental that fits no album.",
    ), "The host's podcast has 20,000 listeners and credits the music with a link.",
        "What can I do with the instrumental that doesn't fit any album?",
        "Offer the instrumental to the podcast host as theme music, for the listeners and the credit link."),
    _t("music-stream-drop", "reversal", _MUSIC, "the new song's streams", (
        "The new song's streams dropped sharply after the third day.",
    ), "The playlist that carried it rotates songs weekly on Thursdays, and streams have held steady "
       "at the organic level since.",
        "Is the new song flopping?",
        "Not a flop: the playlist feature ended, and streams settled at the organic level."),
    _t("music-guitarist-silent", "decoy", _MUSIC, "the guitarist's silence", (
        "The lead guitarist hasn't replied to my messages for five days.",
        "There have been no messages from him in the band chat either.",
    ), "He told the band he'd be away on a trip without his phone until Sunday.",
        "The guitarist has gone quiet. Should I look for a replacement?",
        "Not a problem: he said he'd be away without his phone until Sunday."),
)

TEMPLATES: Final[dict[Split, tuple[Template, ...]]] = {"development": _DEVELOPMENT, "held_out": _HELD_OUT}


def fields(split: Split) -> tuple[str, ...]:
    """The split's fields, in order."""
    return tuple(dict.fromkeys(template.field for template in TEMPLATES[split]))


def templates_for(split: Split, field: str) -> tuple[Template, ...]:
    """The split's templates in ``field``; empty for a field the split doesn't hold."""
    return tuple(template for template in TEMPLATES[split] if template.field == field)
