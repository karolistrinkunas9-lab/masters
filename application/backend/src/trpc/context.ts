import { PrismaService } from '../prisma/prisma.service';
import { HuggingFaceService } from '../services/huggingface.service';

export interface Context {
  prisma: PrismaService;
  huggingFace: HuggingFaceService;
}

export const createContext = (prisma: PrismaService, huggingFace: HuggingFaceService): Context => {
  return {
    prisma,
    huggingFace,
  };
};

