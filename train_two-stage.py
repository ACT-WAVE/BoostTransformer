import random
from torch.backends import cudnn

from data import ImageField, TextField, RawField
from data import DataLoader, Sydney, UCM, RSICD
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer import Transformer, BoostEncoder, BoostMeshedDecoder
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile

import warnings
warnings.filterwarnings("ignore")



def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_torch()


def evaluate_loss(model, dataloader, loss_fn, text_field):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (detections, captions) in enumerate(dataloader):
                detections, captions = detections.to(device), captions.to(device)
                out = model(detections, captions, isencoder=True)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field):
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(dataloader):
            detections = images.to(device)

            with torch.no_grad():
                out, _ = model.beam_search(detections, 20, text_field.vocab.stoi['<eos>'],
                                           5, out_size=1, freeze_backbone=False)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, optim, text_field, freeze_backbone=False):
    # Training with cross-entropy
    model.train()
    scheduler.step()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, captions) in enumerate(dataloader):
            detections, captions = detections.to(device), captions.to(device)

            if not freeze_backbone:
                out = model(detections, captions, isencoder=True, use_DropKey=True)
            else:
                model.encoder.eval()
                with torch.no_grad():
                    encoder_outputs = model.encoder(detections, isencoder=True, use_DropKey=False)
                out = model.decoder(captions, *encoder_outputs)

            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            scheduler.step()

    loss = running_loss / len(dataloader)
    return loss


def train_scst(model, dataloader, optim, cider, text_field, freeze):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    model.train()
    running_loss = .0
    seq_len = 20
    beam_size = 5

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(dataloader):
            detections = images.to(device)

            outs, log_probs = model.beam_search(detections, 20, text_field.vocab.stoi['<eos>'],
                                                beam_size, out_size=beam_size, freeze_backbone=freeze)

            optim.zero_grad()

            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline



if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='BoostTransformer')
    parser.add_argument('--exp_name', type=str, default='UCM', help='Sydney, UCM, RSICD')
    parser.add_argument('--annotation_folder', type=str,
                        default='/root/autodl-tmp/captioning/datasets/UCM_Captions')
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')

    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--N_decoders', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--learning_rate', type=int, default=5e-2)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--RL_lr_ratio', type=int, default=1e-4)
    parser.add_argument('--RL_freeze', action='store_false', default=True, 
                        help='mention to disable RL_freeze, or default enable')
    
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--resume_from', type=int, default=-1)
    parser.add_argument('--comment', type=str, default='None')

    args = parser.parse_args()
    print(args)
    print('Transformer Training')

    # 日志
    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))
    writer.add_text('Info', str(vars(args)), 0)

    # Pipeline for image
    image_field_train = ImageField(origin_img_path=args.annotation_folder+'/imgs', train=True)
    image_field_valtest = ImageField(origin_img_path=args.annotation_folder+'/imgs', train=False)
    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    if args.exp_name == 'Sydney':
        RL_thresh = 2
        index = list(range(613))
        np.random.shuffle(index)
        index_split = [index[:497], index[497:555], index[555:613]]
        dataset_train = Sydney(image_field_train, index_split,
                               text_field, 'Sydney/images/', args.annotation_folder, args.annotation_folder)
        dataset_valtest = Sydney(image_field_valtest, index_split,
                                 text_field, 'Sydney/images/', args.annotation_folder, args.annotation_folder)
    
    elif args.exp_name == 'UCM':
        RL_thresh = 2
        index = list(range(2100))
        np.random.shuffle(index)
        index_split = [index[:1680], index[1680:1890], index[1890:2100]]
        dataset_train = UCM(image_field_train, index_split,
                            text_field, 'UCM/images/', args.annotation_folder, args.annotation_folder)
        dataset_valtest = UCM(image_field_valtest, index_split,
                              text_field, 'UCM/images/', args.annotation_folder, args.annotation_folder)
        
    elif args.exp_name == 'RSICD':
        RL_thresh = 1.5
        index = list(range(10921))
        np.random.shuffle(index)
        index_split = [index[:8734], index[8734:9828], index[9828:10921]]
        dataset_train = RSICD(image_field_train, index_split,
                              text_field, 'RSICD/images/', args.annotation_folder, args.annotation_folder)
        dataset_valtest = RSICD(image_field_valtest, index_split,
                                text_field, 'RSICD/images/', args.annotation_folder, args.annotation_folder)
    
    train_dataset, _, _ = dataset_train.splits
    _, val_dataset, test_dataset = dataset_valtest.splits

    if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
        text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))

    encoder = BoostEncoder(0)
    decoder = BoostMeshedDecoder(len(text_field.vocab), 127, args.N_decoders, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    dict_dataset_train = train_dataset.image_dictionary({'image': image_field_train, 'text': RawField()})
    ref_caps_train = list(train_dataset.text)
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field_valtest, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field_valtest, 'text': RawField()})


    def lambda_lr(s):
        warm_up = args.warmup
        s += 1
        return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)

    # Initial conditions
    optim = Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    Training_Mode = 1
    best_cider = .0
    best_test_cider = .0
    patience = 0
    start_epoch = 0


    if args.resume_last or args.resume_best or (args.resume_from > 0):
        if args.resume_last:
            fname = 'saved_models/%s_last.pth' % args.exp_name
        elif args.resume_best:
            fname = 'saved_models/%s_best.pth' % args.exp_name
        else:
            fname = 'saved_models/%s_%s.pth' % (args.exp_name, args.resume_from)

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            best_test_cider = data['best_test_cider']
            patience = data['patience']
            Training_Mode = data['Training_Mode']

            model.load_state_dict(data['state_dict'], strict=False)
            if Training_Mode > 1 and args.RL_freeze:
                for param in model.encoder.parameters():
                    param.requires_grad = False
                model.encoder.eval()
                torch.cuda.empty_cache()
                optim = Adam(model.decoder.parameters())
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])


    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=args.workers, drop_last=True)
    dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                                num_workers=args.workers)
    dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, shuffle=True,
                                        num_workers=args.workers)
    dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)

    print("Training starts")
    
    for e in range(start_epoch, start_epoch + args.epochs):
        if Training_Mode == 1:
            train_loss = train_xe(model, dataloader_train, optim, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
        elif Training_Mode == 2:
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim, 
                                                             cider_train, text_field, args.RL_freeze)
            writer.add_scalar('data/train_loss', train_loss, e)
            writer.add_scalar('data/reward', reward, e)
            writer.add_scalar('data/reward_baseline', reward_baseline, e)

        # Validation loss
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field)
        writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, text_field)
        print("Validation scores", scores)
        val_cider = scores['CIDEr']
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu2', scores['BLEU'][1], e)
        writer.add_scalar('data/val_bleu3', scores['BLEU'][2], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)
        writer.add_scalar('data/val_spice', scores['SPICE'], e)
        writer.add_scalar('data/val_S*',
                          (scores['BLEU'][3] + scores['METEOR'] + scores['ROUGE'] + scores['CIDEr']) / 4, e)
        writer.add_scalar('data/val_Sm', (
                scores['BLEU'][3] + scores['METEOR'] + scores['ROUGE'] + scores['CIDEr'] + scores['SPICE']) / 5, e)

        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1
            best = False
        print('    patience: ' + str(patience))
        writer.add_scalar('data/patience', patience, e)


        # Test scores
        scores = evaluate_metrics(model, dict_dataloader_test, text_field)
        test_cider = scores['CIDEr']
        writer.add_scalar('data/test_cider', scores['CIDEr'], e)
        writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu2', scores['BLEU'][1], e)
        writer.add_scalar('data/test_bleu3', scores['BLEU'][2], e)
        writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', scores['ROUGE'], e)
        writer.add_scalar('data/test_spice', scores['SPICE'], e)
        writer.add_scalar('data/test_S*',
                          (scores['BLEU'][3] + scores['METEOR'] + scores['ROUGE'] + scores['CIDEr']) / 4, e)
        writer.add_scalar('data/test_Sm', (
                scores['BLEU'][3] + scores['METEOR'] + scores['ROUGE'] + scores['CIDEr'] + scores['SPICE']) / 5, e)

        if test_cider >= best_test_cider:
            best_test_cider = test_cider
            best_test = True
        else:
            best_test = False


        # Prepare for next epoch
        switch = False
        exit_train = False
        if best_cider > RL_thresh and patience >= 10:
            Training_Mode += 1
            patience = 0
            if Training_Mode == 2:
                switch = True
                print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True


        if switch and not best:
            data = torch.load('saved_models/%s_best.pth' % args.exp_name)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            # 先加载权重
            model.load_state_dict(data['state_dict'])
            # 再冻结参数
            if args.RL_freeze:
                for param in model.encoder.parameters():
                    param.requires_grad = False
                model.encoder.eval()
                torch.cuda.empty_cache()
                # 最后创建新优化器
                optim = Adam(model.decoder.parameters(), lr=args.learning_rate * args.RL_lr_ratio)
            else:
                optim = Adam(model.parameters(), lr=args.learning_rate * args.RL_lr_ratio)


        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'best_test_cider': best_test_cider,
            'Training_Mode': Training_Mode,
        }, 'saved_models/%s_last.pth' % args.exp_name)
        print("Test scores", scores)
        if best:
            copyfile('saved_models/%s_last.pth' % args.exp_name, 'saved_models/%s_best.pth' % args.exp_name)

        if best_test:
            copyfile('saved_models/%s_last.pth' % args.exp_name, 'saved_models/%s_best_test.pth' % args.exp_name)

         #save every 50 epochs
        if e > 0 and e % 50 == 0:
            copyfile('saved_models/%s_last.pth' % args.exp_name, 'saved_models/%s_%s.pth' % (args.exp_name, e))

        if exit_train:
            writer.close()
            break
