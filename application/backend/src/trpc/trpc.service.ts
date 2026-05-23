import { Injectable } from '@nestjs/common';
import { appRouter } from './app.router';
import { PrismaService } from '../prisma/prisma.service';
import { HuggingFaceService } from '../services/huggingface.service';
import { createContext } from './context';

@Injectable()
export class TrpcService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly huggingFace: HuggingFaceService,
  ) {}

  getRouter() {
    return appRouter;
  }

  createCaller() {
    return appRouter.createCaller(createContext(this.prisma, this.huggingFace));
  }

  getContext() {
    return createContext(this.prisma, this.huggingFace);
  }
}

