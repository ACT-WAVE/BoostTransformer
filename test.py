import random
from data import ImageField, TextField, RawField
from data import DataLoader, Sydney, UCM, RSICD
import evaluation
from models.transformer import Transformer, BoostEncoder, BoostMeshedDecoder
import torch
from tqdm import tqdm
import argparse, os, pickle
import numpy as np
import itertools
import warnings
warnings.filterwarnings("ignore")


def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
seed_torch()


def predict_captions(model, dataloader, text_field):
    model.eval()
    gen, gts = {}, {}
    eos_idx = text_field.vocab.stoi['<eos>']
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(dataloader):
            detections = images.to(device, non_blocking=True)

            with torch.no_grad():
                out, _ = model.beam_search(detections, 20, eos_idx, 
                                           5, out_size=1, freeze_backbone=False)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                key = f"{it}_{i}"
                gen[key], gts[key] = [gen_i, ], gts_i
            pbar.update()
            
    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores



if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='BoostTransformer')
    parser.add_argument('--exp_name', type=str, default='UCM', help='Sydney, UCM, RSICD')
    parser.add_argument('--annotation_folder', type=str,
                        default='/root/autodl-tmp/captioning/datasets/UCM_Captions')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=0)

    args = parser.parse_args()
    print(args)
    print('Transformer Evaluation')

    # Pipeline for image
    image_field = ImageField(origin_img_path=args.annotation_folder+'/imgs', train=False)
    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    if args.exp_name == 'Sydney':
        index = list(range(613))
        np.random.shuffle(index)
        index_split = [index[:497], index[497:555], index[555:613]]
        dataset_valtest = Sydney(image_field, index_split,
                                 text_field, 'Sydney/images/', args.annotation_folder, args.annotation_folder)
    
    elif args.exp_name == 'UCM':
        index = list(range(2100))
        np.random.shuffle(index)
        index_split = [index[:1680], index[1680:1890], index[1890:2100]]
        dataset_valtest = UCM(image_field, index_split,
                              text_field, 'UCM/images/', args.annotation_folder, args.annotation_folder)
        
    elif args.exp_name == 'RSICD':
        index = list(range(10921))
        np.random.shuffle(index)
        index_split = [index[:8734], index[8734:9828], index[9828:10921]]
        dataset_valtest = RSICD(image_field, index_split,
                                text_field, 'RSICD/images/', args.annotation_folder, args.annotation_folder)

    _, _, test_dataset = dataset_valtest.splits
    text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))


    # Model and dataloaders
    encoder = BoostEncoder(0)
    decoder = BoostMeshedDecoder(len(text_field.vocab), 127, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    data = torch.load("./saved_models/UCM_best.pth")

    model.load_state_dict(data['state_dict'])
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    
    scores = predict_captions(model, dict_dataloader_test, text_field)
    print(scores)
