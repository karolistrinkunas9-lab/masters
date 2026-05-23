import { router } from './trpc';
import { requirementsRouter } from './routers/requirements.router';

export const appRouter = router({
  requirements: requirementsRouter,
});

export type AppRouter = typeof appRouter;

