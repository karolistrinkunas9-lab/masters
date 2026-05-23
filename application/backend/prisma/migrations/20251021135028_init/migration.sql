-- CreateTable
CREATE TABLE "requirements" (
    "id" TEXT NOT NULL,
    "text" TEXT NOT NULL,
    "quality" TEXT NOT NULL DEFAULT 'Medium',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "requirements_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "requirement_modifications" (
    "id" TEXT NOT NULL,
    "requirementId" TEXT NOT NULL,
    "stakeholderFeedback" TEXT,
    "modifiedText" TEXT NOT NULL,
    "aiModel" TEXT NOT NULL,
    "detectedSubtypes" TEXT[],
    "qualityOfModification" TEXT NOT NULL,
    "correctnessOfChange" TEXT NOT NULL,
    "aiModelComment" TEXT,
    "reqQuality" TEXT NOT NULL,
    "changeQuality" TEXT NOT NULL,
    "finalResult" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "requirement_modifications_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "requirement_modifications" ADD CONSTRAINT "requirement_modifications_requirementId_fkey" FOREIGN KEY ("requirementId") REFERENCES "requirements"("id") ON DELETE CASCADE ON UPDATE CASCADE;
