import json
import re

classes = [
    'ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching', 'BaseballPitch',
    'Basketball', 'BasketballDunk', 'BenchPress', 'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles',
    'BodyWeightSquats', 'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth',
    'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen', 'Diving', 'Drumming',
    'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', 'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut',
    'Hammering', 'HammerThrow', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump', 'HorseRace',
    'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow', 'JugglingBalls', 'JumpRope', 'JumpingJack',
    'Kayaking', 'Knitting', 'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', 'Nunchucks',
    'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute', 'PlayingGuitar',
    'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps',
    'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'ShavingBeard',
    'Shotput', 'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty', 'StillRings',
    'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus',
    'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups',
    'WritingOnBoard', 'YoYo'
]

templates = [
    'a photo of a person {}.',
    'a video of a person {}.',
    'a example of {}.',
    'a demonstration of {}.',
    'a video of {}.',
    'a photo of {}.',
    'the person is {}.',
    'doing {}.'
]

def clean_label(label):
    # CamelCase to space separated
    return re.sub('([a-z])([A-Z])', r'\1 \2', label).lower()

prompts_dict = {}
for cls in classes:
    cleaned = clean_label(cls)
    # Special cases for better grammar if desired, but ActionCLIP usually stays simple.
    # We'll stick to clean space-separated labels as per ActionCLIP.
    prompts_dict[cls] = [t.format(cleaned) for t in templates]

with open('ucf101_actionclip_prompts.json', 'w') as f:
    json.dump(prompts_dict, f, indent=4)

print("Generated ucf101_actionclip_prompts.json")
