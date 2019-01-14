from PIL import Image, ImageDraw

im = Image.new('RGB', (args.width, args.height), color='red')
draw = ImageDraw.Draw(im)
